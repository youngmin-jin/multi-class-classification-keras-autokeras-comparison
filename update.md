# 141023
Jin) 
- The number of <ins>epochs has been changed from 300 to 100</ins> due to consuming time resulting from the expanded grid. This change has been applied to all the models to maintain consistency. The results and codes of the <ins>1 and 2 models have been uploaded</ins>, and the 3 models' will be updated as soon as the results come out. 
- The 2 models have been run with the expanded grid (neurons=[100,500,1000,1500,2000])
- According to the results of the 2 models, <ins>the expanded grid does not significantly impact on the better performance</ins> as they showed the similar or lower scores even though the best parameters of number of neurons are higher.
- <ins>The i2 without data augmentation worked fine</ins> in the test running. I am planning to apply data augmentation to the i3 after getting the results of 2 models.

<br/>

## The original result and best hyperparameters (in the dissertation)<br/>
  > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/5f88408a-13fc-4942-9130-14bc750f3313) <br/><br/>
<br/>

## Structured data models
### s1: <ins>completed</ins>
  - running time
    > Slurm Job_id=209194 Name=s1.slurm Ended, Run time **00:00:55**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/0ea1b0c2-8c16-41da-9b60-ac38c910970d) <br/>
  - The full result can be confirmed here [https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/1#issue-1920581046](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/4#issue-1939208249) <br/><br/>

### s2: <ins>completed</ins> (kerastuner 1.3.5)
  - running time
    >  Slurm Job_id=210579 Name=s2.slurm Ended, Run time **01:57:11**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/3452e887-516f-4e9d-92d1-5a41785ce61d) <br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/a908f2ed-3dc0-4bf4-ab30-65fef5f30de1) <br/>
  - best parameter
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/f7ac3e08-7fda-4e21-927c-882dac2276b0)
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/7#issue-1941048499 <br/><br/>

### s3: <ins>waiting for running</ins> (after finishing the 2 models) due to the conflict of autokeras and kerastuner version

<br/><br/>

## Text data models
### t1: <ins>completed</ins>
  - running time
    >  Slurm Job_id=209201 Name=t1.slurm Ended, Run time **00:01:16**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/605d153d-d189-44e1-92d3-e844a509aa7a) <br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/ed7b85e9-faf2-42e6-b87a-bea1382283ee) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/5#issue-1939208665 <br/><br/>

### t2: <ins>completed</ins> (kerastuner 1.3.5)
  - running time
    >  Slurm Job_id=216970 Name=t2.slurm Ended, Run time **04:38:05**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/e727f55a-2989-450a-a27d-6684b5c34cff) <br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/82d118fe-ce3e-4f0f-8f96-979e5d47d167) <br/>
  - best parameter
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/8cc3698b-f4b5-45bb-8679-b02d7c818fc3)
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/8#issue-1942777984 <br/><br/>

### t3: <ins>waiting for running</ins> (after finishing the 2 models) due to the conflict of autokeras and kerastuner version

<br/><br/>

## Image data models
### i1: <ins>completed</ins>
  - running time
    >  Slurm Job_id=209202 Name=i1.slurm Ended, Run time **00:09:00**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/60564f5c-98f7-4146-a36f-3bdcae9cf50a) <br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/623c71b0-d184-439c-8908-875c06f1b9ac) <br/>
  - The full result can be confirmed here [https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/5#issue-1939208665](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/6#issue-1939208864) <br/><br/>

### i2: <ins>completed</ins> (kerastuner 1.3.5)
  - running time
    > Slurm Job_id=209205 Name=i2.slurm Ended, Run time 2- **09:09:00** (I assumed more than 9h, but it appears like this)
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/a7b69e2d-e3a3-4d06-9acd-d49271bc97e1) <br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/99a6acd0-eb6b-4dbd-ba71-551bfe76b16e) <br/>
  - best parameters
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/f94e58b6-ddff-4165-b56d-296698219539) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/9#issue-1943605147 <br/><br/>

### i3: <ins>running with data augmentation</ins>





