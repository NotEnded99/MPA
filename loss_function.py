import torch
import torch.nn as nn

def compute_attack_loss(next_state_pred, args):
    s1 = next_state_pred.squeeze(0)[1]  
    s0 = next_state_pred.squeeze(0)[0] 
    loss = 0.0
    lambda_p = 2.0  

    if args.env_id == "Hopper-v3":
        if args.attack_flag == "a_up":  
            loss += torch.relu(0.21 - s1)  
            loss += lambda_p*torch.relu(-0.15-s1) 
            loss += lambda_p*torch.relu(1.0-s0)  

        elif args.attack_flag == "a_low": 
            loss += torch.relu(s1 + 0.21)  
            loss += lambda_p*torch.relu(s1 - 0.15)  
            loss += lambda_p*torch.relu(1.0-s0)   

        elif args.attack_flag == "z_low":  
            loss += torch.relu(s0 - 0.7) 
            loss += lambda_p*torch.relu(-s1 - 0.12)  
            loss += lambda_p*torch.relu(s1 - 0.12)   
        else:
            raise ValueError("Invalid attack flag. Use 'a_up', 'a_low', or 'z_low'.")
        
    elif args.env_id == "Walker2d-v3":

        if args.attack_flag == "a_up":  
            loss += torch.relu(1.05 - s1)   
            loss += lambda_p*torch.relu(-0.7-s1)  
            loss += lambda_p*torch.relu(1.0-s0)   

        elif args.attack_flag == "a_low":  
            loss += torch.relu(s1 + 1.05)  
            loss += lambda_p*torch.relu(s1 - 0.7)   
            loss += lambda_p*torch.relu(1.0-s0)  

        elif args.attack_flag == "z_low":  
            loss += torch.relu(s0-0.75) 
            loss += lambda_p*torch.relu(-0.7-s1)  
            loss += lambda_p*torch.relu(s1 - 0.7)  
        else:
            raise ValueError("Invalid attack flag. Use 'a_up', 'a_low', or 'z_low'.")
    
    elif args.env_id == "Ant-v3":
        if args.attack_flag == "z_up":  
            loss += torch.relu(1.05-s0)  
        else:
            raise ValueError("Invalid attack flag. Use 'z_up'.")

    elif args.env_id == "HalfCheetah-v3":
        if  args.attack_flag == "z_low":  
            loss += torch.relu(s0+0.4) 
        else:
            raise ValueError("Invalid attack flag. Use 'z_low'.")
    return loss
