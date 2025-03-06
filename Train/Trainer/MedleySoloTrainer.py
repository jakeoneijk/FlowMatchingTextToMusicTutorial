#reference
import torch

from TorchJaekwon.Train.Trainer.Trainer import Trainer, TrainState
from TorchJaekwon.Util import UtilTorch

class MedleySoloTrainer(Trainer):
    def run_step(self,data,metric,train_state:TrainState):
        data = data.to(self.device)
        loss = self.model( data , is_cond_unpack = True)
        metric = self.update_metric(metric, {'loss': loss}, data.shape[0])
        return loss, metric
    
    @torch.no_grad()
    def log_media(self) -> None:
        gen_audio_num:int = 8
        x_shape:tuple = self.model.get_x_shape(None)
        x_shape = ( gen_audio_num, *x_shape[1:] )
        gen_audio = self.model.infer(x_shape = x_shape)
        for i in range(gen_audio_num):
            self.log_writer.plot_audio(name = f'test_{i}', audio_dict = { 'gen': UtilTorch.to_np(gen_audio[i][0]) }, global_step=self.global_step, sample_rate=44100)

