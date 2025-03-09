#reference
import torch

from TorchJaekwon.Train.Trainer.Trainer import Trainer, TrainState
from TorchJaekwon.Util import UtilTorch

class MedleySoloTrainer(Trainer):
    def run_step(self,data,metric,train_state:TrainState):
        data = data.to(self.device)
        loss = self.model( data, cond = {'audio': data} )
        metric = self.update_metric(metric, {'loss': loss}, data.shape[0])
        return loss, metric
    
    @torch.no_grad()
    def log_media(self) -> None:
        text_list = [
            'clarinet',
            'electric guitar',
            'singer',
            'flute',
            'piano',
            'saxophone',
            'trumpet',
            'violin'
        ]
        gen_audio = self.model.infer( cond = {'text': text_list})
        for i in range(len(text_list)):
            self.log_writer.plot_audio(name = text_list[i], audio_dict = { 'gen': UtilTorch.to_np(gen_audio[i][0]) }, global_step=self.global_step, sample_rate=44100)

