
Meta Automatic Curriculum Learning
==================================

## Important: visit https://sites.google.com/view/meta-acl for videos !

## The AGAIN algorithm is implemented in teachDRL/teachers/again.py

## Installation

1- Unzip and install codebase with Python >= 3.6, using Conda for example 
```
cd meta-ACL/
conda create --name metaACL python=3.6
conda activate metaACL
pip install -e .
```

## Launching experiments on the toy env
2 versions of the toy environment can be used.

### 4 possible student types (example is with ALP-GMM teacher)
```
python3 toy_run.py --seed 42 --exp_name test_toy_env --teacher ALP-GMM -rnd 10
```
### all 400 possible student types (example is with ALP-GMM teacher)
```
python3 toy_run.py --seed 42 --exp_name test_full_toy_env -v2 --teacher ALP-GMM -rnd 10 -rsc
```
## Launching experiments on the Parkour env

### Test the environment & student space
```
python3 vizu_walker_climber_environment.py
```

### Example of launching student training with ALP-GMM teacher
```
python3 run.py --seed 42 --exp_name test_parkour --teacher ALP-GMM -rnd 10 --nb_test_episodes 225 -rndls
```

## Launching AGAIN
To test AGAIN, one first has to train a batch of students with ALP-GMM (i.e. a classroom).
The resulting runs' data (located in teachDRL/data/<your-alpgmm-exp-name>/) then has to be processed
to generate a classroom file that will be given to AGAIN (with the -cf argument).

Here is the process detailed for the regular toy environment:
#### Step 0 - Train a batch of students with ALP-GMM
```
python3 toy_run.py --seed 0 --exp_name test_toy_env --teacher ALP-GMM -rnd 10
python3 toy_run.py --seed 1 --exp_name test_toy_env --teacher ALP-GMM -rnd 10
python3 toy_run.py --seed 2 --exp_name test_toy_env --teacher ALP-GMM -rnd 10
...
```
#### Step 1 - Process data to create classroom file
Use toy_env_classroom_maker.ipynb to generate the classroom (i.e. toy_classroom.pkl)
You also need to move your classroom data to a new folder:
```
mv teachDRL/data/<your-alpgmm-exp-name>/ teachDRL/data/elders_knowledge/
```
#### Step 2 - Launch AGAIN
```
python3 toy_run.py --seed 42 --exp_name test_metaACL_toy_env --teacher AGAIN --expert_type R --use_alpgmm -rnd 2 --nb_cubes 20 -pt 2 -sR -cf toy_classroom
```




Aknowledgment: Our approach and our environments are implemented on top of two main codebases:

https://github.com/openai/spinningup

https://github.com/flowersteam/teachDeepRL

Many thanks to their respective authors.
