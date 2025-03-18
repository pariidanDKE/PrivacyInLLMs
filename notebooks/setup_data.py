import os
import hydra
from omegaconf import OmegaConf


os.environ["DATASPLIT"] = "forget05_perturbed"
os.environ["OUTPUTMODELDIR"] = "output_directory"
os.environ["HF_HOME"] = f"/scratch-shared/{os.environ['USER']}/hf-cache-dir" 
os.environ["HF_TOKEN"] = "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"


hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path="configs")

cfg = hydra.compose(config_name="tune_config")
cfg.data_mode.retain_num = 40

print(OmegaConf.to_yaml(cfg))


cfg.data.dataset.split = "forget01_perturbed" # if forgot to declare $DATASETSPLIT

data_module = create_datamod(
    dataset_config=cfg.data.dataset,
    conv_template_config=cfg.data.conv_template,
    data_mode_config=cfg.data_mode,
    tokenizer=tokenizer,
)
data_module.prepare_data()