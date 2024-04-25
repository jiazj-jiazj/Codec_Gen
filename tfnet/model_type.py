def is_s2s(model_type):
    if 'mbart' in model_type or 's2s' in model_type:
        return True
    return False

def is_vqvae(config):
    return True if "vqvae" in config["model_type"] else False
    
def is_controllable_dns_vqvae(config):
    return True if "controldns" in config["model_type"] else False
    
def is_multirate_vqvae(config):
    return True if config['model_type'] == 'tfnetv2_interleave_multiraterps_vqvae' and (config['bitrate'] == 'all4') else False
    
def is_bitrate_scalable(config):
    return True if '_multidec' in config['model_type'] else False