# 050224
Jin) 
- All models appear to be re-run using at least two GPUs (To2 model might require more). The re-running of the To2 model is in progress to clarify if it doesn't work with two GPUs.
- Io2 model acheived the highest accuracy of 0.93, indicating that further finetuning is required.
<br/>


# $${\color{red}Legacy \space model \space names \space below \space (These \space will \space be \space updated \space soon)}$$
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

### t2-b 
**1) when applying <ins>'bert_base_en_uncased'</ins> with all dropout, LR, batchsize range used, epochs=5** (kerastuner 1.3.5)
  - running time
    > Slurm Job_id=611888 Name=t2-b_new.slurm Ended, Run time **05:48:29**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/fd8679b2-cae0-436b-8ae8-64755b74aa6a) <br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/c4c1e6f9-f1ef-47a4-8f83-8a84b91927f8)
  - best parameter
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/5362e68b-a3e9-4174-a6ba-969ab501f241)
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/23#issue-2027595019 <br/><br/>

**2) when applying <ins>'bert_large_en_uncased'</ins> with all dropout, LR, batchsize range used, epochs=5** (kerastuner 1.3.5)
  - running time
    > Slurm Job_id=611887 Name=t2-b_new.slurm Ended, Run time **05:48:18**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/92fabf43-acd8-4ba8-af5e-e0f2b67912c4)<br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/79403f6d-f3c9-41e4-8ed0-dc0346223e4d)
  - best parameter
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/99a8ea7d-30da-4beb-ac95-51b549a3717b)
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/24#issue-2027595343 <br/><br/>

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





