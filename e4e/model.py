from Pixel2Style2Pixel.submodel.discriminator import LatentCodesDiscriminator
import torch
import math
from lib import utils
from lib.model_interface import ModelInterface
from e4e.loss import E4ELoss
from e4e.nets import E4EEncoder
from lib.discriminators import LatentCodesDiscriminator


class E4EModel(ModelInterface):
    def set_networks(self):
        self.setup_progressive_steps()
        self.G = E4EEncoder().cuda(self.gpu).train()
        self.D = LatentCodesDiscriminator().cuda(self.gpu).train()

    def setup_progressive_steps(self):
        self.G.progressive_stage = 0
        log_size = int(math.log(1024, 2))
        num_style_layers = 2*log_size - 2 # 18
        num_deltas = num_style_layers - 1 # 17

        self.progressive_steps = [0]
        next_progressive_step = self.args.progressive_start
        for i in range(num_deltas):
            self.progressive_steps.append(next_progressive_step)
            next_progressive_step += self.args.progressive_step_cycle

    def update_progressive_stage(self, step):
        if step in self.progressive_steps:
            progressive_stage = self.progressive_steps.index(step)
            self.G.progressive_stage = progressive_stage
            print(f"==============================================================")
            print(f">>> progressive_stage is converted to stage {progressive_stage}")
            print(f"==============================================================")

    def set_loss_collector(self):
        self._loss_collector = E4ELoss(self.args)

    def go_step(self, global_step):
        # load batch
        
        self.update_progressive_stage(global_step)

        I_source = self.load_next_batch()
        self.dict["I_source"] = I_source

        # run G
        self.run_G()

        # update G
        loss_G = self.loss_collector.get_loss_G(self.dict)
        utils.update_net(self.opt_G, loss_G)

        # run D
        loss_D = self.run_D()
        utils.update_net(self.opt_D, loss_D)

        # update D
        self.loss_collector.get_loss_D(self.dict)

        # print images
        self.train_images = [
            self.dict["I_source"], 
            self.dict["I_recon"],
            ]

    def run_G(self):
        I_recon, w_fake = self.G(self.dict["I_source"])
        d_adv = self.D(w_fake)
        id_recon = self.G.get_id(I_recon)
        with torch.no_grad():
            id_source = self.G.get_id(self.dict["I_source"]).detach()

        self.dict["I_recon"] = I_recon
        self.dict["id_source"] = id_source
        self.dict["id_recon"] = id_recon
        self.dict["d_adv"] = d_adv

    def run_D(self):
        d_real= self.D(self.dict["w_real"], None)
        d_fake= self.D(self.dict["w_fake"].detach(), None)

        self.dict["d_real"] = d_real
        self.dict["d_fake"] = d_fake

    def do_validation(self, step):
        with torch.no_grad():
            result_images = self.G(self.valid_source, self.valid_target)[0]
        self.valid_images = [
            self.valid_source, 
            self.valid_target, 
            result_images
            ]

    @property
    def loss_collector(self):
        return self._loss_collector
        