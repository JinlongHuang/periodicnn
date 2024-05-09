from dotenv import load_dotenv
import os
import wandb
import json
import exp

if __name__ == "__main__":
    # with open("config.json", 'r') as f:
    #     config = json.load(f)
    # use_wandb = config['analysis']['use_wandb']

    # if use_wandb:
    #     load_dotenv('wandb_api.env')
    #     wandb_api = os.getenv('WANDB_API')
    #     wandb.login(key=wandb_api)

    # for i in range(1, 10):
    #     config["train"]["torch_seed"] = i
    #     with open("config.json", 'w') as f:
    #         json.dump(config, f, indent=4)

    #     print(f"Running experiment with seed = {i} ...")
    #     exp.run_crypto(i, use_wandb)

    exp.run_monash()
