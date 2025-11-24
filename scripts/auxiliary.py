import wandb

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def setup_wandb(cfg_run, cfg_export):
    run = wandb.init(
        project= cfg_export['project_name'],
        name= cfg_export.get('run_name', 'default'),
        config= cfg_run,
        job_type= 'training'
    )

    return run

