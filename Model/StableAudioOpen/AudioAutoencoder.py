import torch
import torch.nn as nn

from Model.StableAudioOpen.AutoencoderModule import OobleckEncoder, OobleckDecoder
from Model.StableAudioOpen.AutoencoderModule import VAEBottleneck, Bottleneck

class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder = OobleckEncoder(),
        decoder = OobleckDecoder(),
        latent_dim = 64,
        downsampling_ratio = 2048,
        sample_rate = 44100,
        io_channels=2,
        bottleneck: Bottleneck = VAEBottleneck(),
        pretransform = None,
        in_channels = None,
        out_channels = None,
        soft_clip = False
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels
        self.out_channels = io_channels

        self.min_length = self.downsampling_ratio

        if in_channels is not None:
            self.in_channels = in_channels

        if out_channels is not None:
            self.out_channels = out_channels

        self.bottleneck = bottleneck

        self.encoder = encoder

        self.decoder = decoder

        self.pretransform = pretransform

        self.soft_clip = soft_clip
 
        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

    def encode(self, audio, return_info=False, skip_pretransform=False, iterate_batch=False, **kwargs):

        info = {}

        if self.pretransform is not None and not skip_pretransform:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    audios = []
                    for i in range(audio.shape[0]):
                        audios.append(self.pretransform.encode(audio[i:i+1]))
                    audio = torch.cat(audios, dim=0)
                else:
                    audio = self.pretransform.encode(audio)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        audios = []
                        for i in range(audio.shape[0]):
                            audios.append(self.pretransform.encode(audio[i:i+1]))
                        audio = torch.cat(audios, dim=0)
                    else:
                        audio = self.pretransform.encode(audio)

        if self.encoder is not None:
            if iterate_batch:
                latents = []
                for i in range(audio.shape[0]):
                    latents.append(self.encoder(audio[i:i+1]))
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.encoder(audio)
        else:
            latents = audio

        if self.bottleneck is not None:
            # TODO: Add iterate batch logic, needs to merge the info dicts
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)

            info.update(bottleneck_info)
        
        if return_info:
            return latents, info

        return latents

    def decode(self, latents, iterate_batch=False, **kwargs):

        if self.bottleneck is not None:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    decoded.append(self.bottleneck.decode(latents[i:i+1]))
                latents = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)

        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                decoded.append(self.decoder(latents[i:i+1]))
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, **kwargs)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    decodeds = []
                    for i in range(decoded.shape[0]):
                        decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                    decoded = torch.cat(decodeds, dim=0)
                else:
                    decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        decodeds = []
                        for i in range(latents.shape[0]):
                            decodeds.append(self.pretransform.decode(decoded[i:i+1]))
                        decoded = torch.cat(decodeds, dim=0)
                    else:
                        decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)
        
        return decoded
          
    def decode_tokens(self, tokens, **kwargs):
        '''
        Decode discrete tokens to audio
        Only works with discrete autoencoders
        '''

        assert isinstance(self.bottleneck, DiscreteBottleneck), "decode_tokens only works with discrete autoencoders"

        latents = self.bottleneck.decode_tokens(tokens, **kwargs)

        return self.decode(latents, **kwargs)
        
    
    def preprocess_audio_for_encoder(self, audio, in_sr):
        '''
        Preprocess single audio tensor (Channels x Length) to be compatible with the encoder.
        If the model is mono, stereo audio will be converted to mono.
        Audio will be silence-padded to be a multiple of the model's downsampling ratio.
        Audio will be resampled to the model's sample rate. 
        The output will have batch size 1 and be shape (1 x Channels x Length)
        '''
        return self.preprocess_audio_list_for_encoder([audio], [in_sr])

    def preprocess_audio_list_for_encoder(self, audio_list, in_sr_list):
        '''
        Preprocess a [list] of audio (Channels x Length) into a batch tensor to be compatable with the encoder. 
        The audio in that list can be of different lengths and channels. 
        in_sr can be an integer or list. If it's an integer it will be assumed it is the input sample_rate for every audio.
        All audio will be resampled to the model's sample rate. 
        Audio will be silence-padded to the longest length, and further padded to be a multiple of the model's downsampling ratio. 
        If the model is mono, all audio will be converted to mono. 
        The output will be a tensor of shape (Batch x Channels x Length)
        '''
        batch_size = len(audio_list)
        if isinstance(in_sr_list, int):
            in_sr_list = [in_sr_list]*batch_size
        assert len(in_sr_list) == batch_size, "list of sample rates must be the same length of audio_list"
        new_audio = []
        max_length = 0
        # resample & find the max length
        for i in range(batch_size):
            audio = audio_list[i]
            in_sr = in_sr_list[i]
            if len(audio.shape) == 3 and audio.shape[0] == 1:
                # batchsize 1 was given by accident. Just squeeze it.
                audio = audio.squeeze(0)
            elif len(audio.shape) == 1:
                # Mono signal, channel dimension is missing, unsqueeze it in
                audio = audio.unsqueeze(0)
            assert len(audio.shape)==2, "Audio should be shape (Channels x Length) with no batch dimension" 
            # Resample audio
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate).to(audio.device)
                audio = resample_tf(audio)
            new_audio.append(audio)
            if audio.shape[-1] > max_length:
                max_length = audio.shape[-1]
        # Pad every audio to the same length, multiple of model's downsampling ratio
        padded_audio_length = max_length + (self.min_length - (max_length % self.min_length)) % self.min_length
        for i in range(batch_size):
            # Pad it & if necessary, mixdown/duplicate stereo/mono channels to support model
            new_audio[i] = prepare_audio(new_audio[i], in_sr=in_sr, target_sr=in_sr, target_length=padded_audio_length, 
                target_channels=self.in_channels, device=new_audio[i].device).squeeze(0)
        # convert to tensor 
        return torch.stack(new_audio) 

    def encode_audio(self, audio, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Encode the entire audio in parallel
            return self.encode(audio, **kwargs)
        else:
            # CHUNKED ENCODING
            # samples_per_latent is just the downsampling ratio (which is also the upsampling ratio)
            samples_per_latent = self.downsampling_ratio
            total_size = audio.shape[2] # in samples
            batch_size = audio.shape[0]
            chunk_size *= samples_per_latent # converting metric in latents to samples
            overlap *= samples_per_latent # converting metric in latents to samples
            hop_size = chunk_size - overlap
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = audio[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = audio[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # Note: y_size might be a different value from the latent length used in diffusion training
            # because we can encode audio of varying lengths
            # However, the audio should've been padded to a multiple of samples_per_latent by now.
            y_size = total_size // samples_per_latent
            # Create an empty latent, we will populate it with chunks as we encode them
            y_final = torch.zeros((batch_size,self.latent_dim,y_size)).to(audio.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # encode the chunk
                y_chunk = self.encode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size // samples_per_latent
                    t_end = t_start + chunk_size // samples_per_latent
                #  remove the edges of the overlaps
                ol = overlap//samples_per_latent//2
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final
    
    def decode_audio(self, latents, chunked=False, overlap=32, chunk_size=128, **kwargs):
        '''
        Decode latents to audio. 
        If chunked is True, split the latents into chunks of a given maximum size chunk_size, with given overlap, both of which are measured in number of latents. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked audio and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        if not chunked:
            # default behavior. Decode the entire latent in parallel
            return self.decode(latents, **kwargs)
        else:
            # chunked decoding
            hop_size = chunk_size - overlap
            total_size = latents.shape[2]
            batch_size = latents.shape[0]
            chunks = []
            for i in range(0, total_size - chunk_size + 1, hop_size):
                chunk = latents[:,:,i:i+chunk_size]
                chunks.append(chunk)
            if i+chunk_size != total_size:
                # Final chunk
                chunk = latents[:,:,-chunk_size:]
                chunks.append(chunk)
            chunks = torch.stack(chunks)
            num_chunks = chunks.shape[0]
            # samples_per_latent is just the downsampling ratio
            samples_per_latent = self.downsampling_ratio
            # Create an empty waveform, we will populate it with chunks as decode them
            y_size = total_size * samples_per_latent
            y_final = torch.zeros((batch_size,self.out_channels,y_size)).to(latents.device)
            for i in range(num_chunks):
                x_chunk = chunks[i,:]
                # decode the chunk
                y_chunk = self.decode(x_chunk)
                # figure out where to put the audio along the time domain
                if i == num_chunks-1:
                    # final chunk always goes at the end
                    t_end = y_size
                    t_start = t_end - y_chunk.shape[2]
                else:
                    t_start = i * hop_size * samples_per_latent
                    t_end = t_start + chunk_size * samples_per_latent
                #  remove the edges of the overlaps
                ol = (overlap//2) * samples_per_latent
                chunk_start = 0
                chunk_end = y_chunk.shape[2]
                if i > 0:
                    # no overlap for the start of the first chunk
                    t_start += ol
                    chunk_start += ol
                if i < num_chunks-1:
                    # no overlap for the end of the last chunk
                    t_end -= ol
                    chunk_end -= ol
                # paste the chunked audio into our y_final output audio
                y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
            return y_final