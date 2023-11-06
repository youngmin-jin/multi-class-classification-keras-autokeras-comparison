# 061123
Jin) 
- All the models except for t2-b are completed (the results are uploaded below)
- t2-b is planned to be developed using BERT 

<br/>

## The original result and best hyperparameters (in the dissertation)<br/>
  > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/5f88408a-13fc-4942-9130-14bc750f3313) <br/><br/>
<br/>

## Structured data models
### s1: <ins>completed</ins>
  - running time
    > Slurm Job_id=403312 Name=s1.slurm Ended, Run time **00:00:48**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/34607aa9-d5df-4eb1-82e1-dfff86cc821b) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/16#issue-1975549274 <br/><br/>

### s2: <ins>completed</ins> (kerastuner 1.3.5)
  - running time
    >  Slurm Job_id=403358 Name=s2.slurm Ended, Run time **01:56:55**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/af2357fd-e5a0-4cde-af09-156bb8a9dee8) <br/>
  - best parameter
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/f08db9d0-0803-46e7-bb07-f4aac46c4b14)
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/19#issue-1976117946 <br/><br/>

### s3: <ins>completed</ins>
  - running time
    > Slurm Job_id=371734 Name=s3.slurm Ended, Run time **02:57:56**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/9df30d05-6f07-49b5-ae95-46d154a8c4eb) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/13#issue-1975481559 <br/><br/>

<br/><br/>

## Text data models
### t1: <ins>completed</ins>
  - running time
    >  Slurm Job_id=403313 Name=t1.slurm Ended, Run time **00:01:19**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/2133f1e6-1044-4579-8953-ee21b73ae21f)
 <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/17#issue-1975549486 <br/><br/>

### t2-a: <ins>completed</ins> (kerastuner 1.3.5)
  - running time
    > Slurm Job_id=403651 Name=t2-a.slurm Ended, Run time **04:00:58**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/5afdfd16-4840-4874-9618-3872576d1c40) <br/>
  - best parameter
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/1cd8df18-0ee0-4381-b087-2228f3970a91)
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/20#issue-1977641140 <br/><br/>

### t2-b: <ins>will be developed</ins> (kerastuner 1.3.5)
  - running time
    >  
  - results
    >  <br/>
  - best parameter
    > 
  - The full result can be confirmed here <br/><br/>

### t3: <ins>completed</ins> 
  - running time
    >  Slurm Job_id=371757 Name=t3.slurm Ended, Run time **02:41:49**
  - results
    >  ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/3da2ef1a-c5ad-40c4-8f08-0c70cd940efe) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/14#issue-1975481856 <br/><br/>

<br/><br/>

## Image data models
### i1: <ins>completed</ins>
  - running time
    >  Slurm Job_id=403338 Name=i1.slurm Ended, Run time **00:02:42**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/c1d9c0c3-d90e-4ed8-bd52-db24b88ddf40) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/18#issue-1975549673 <br/><br/>

### i2: <ins>completed</ins> (kerastuner 1.3.5)
  - running time
    >  Slurm Job_id=472079 Name=i2.slurm Ended, Run time **15:14:35**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/4c4f08ed-7004-45d2-9b66-91fc60377530) <br/>
    ... <br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/72f5aca4-da6c-45f4-b83c-9f33c8f529b8)
  - best parameters
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/dbbc5ac8-9d14-45e7-83fc-552b0d6fe57a) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/21#issue-1978368310 <br/><br/>

### i3: <ins>completed (with data augmentation)</ins>
  - running time
    > Slurm Job_id=371759 Name=i3.slurm Ended, Run time **14:22:32**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/2aae98e8-5bef-4983-b6cb-2bec5053df95) <br/>
    
    ...

    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/f02f71b9-5985-4879-bc3f-1d19d061ea6a) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/15#issue-1975482018 <br/><br/>





