# waggle-dance
There are two provided brances with similar structure. 

Master - Code that reproduces Campos and Froese (2017)
Twotarget - Code that performs the new experiment described in the thesis

Each branch also comes with two pre trained networks.

Example usage:

Train a network (NOTE: You need to create a logs directory)
> py code/train.py configurations/true_campos.json 

Test a network
> py code/test.py configurations/true_campos.json models/MODEL_NAME

Evaluate performance on random targets
> py code/evaluate.py configurations/true_campos.json models/MODEL_NAME ntrials

View performance on a range of targets
> py code/meandistance.py configurations/true_campos.json models/MODEL_NAME
