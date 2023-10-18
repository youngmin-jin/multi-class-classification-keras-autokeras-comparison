# 181023
Jin) 
- The number of epochs has been changed from 300 to 100 due to consuming time resulting from the expanded grid. This change has been applied to all the models to maintain consistency.
- Regarding 2 models
  - All the 2 models have been run with the expanded grid (neurons=[100,500,1000,1500,2000])
  - The expanded grid does not significantly impact on the better performance as they showed the similar or lower scores even though the best parameters of number of neurons are higher.
- Regarding 3 models
  - The result of the s3 and t3 model have been uploaded/ The i3 model (with data augmentation) is running.
  - **<ins>The t3 result is quite different from the original result, showing the highest performance between all t models.</ins>** The t2 was the most well-performed model of all t models in the original result.
  - **<ins>The test running (i3 using another multi-class image dataset) has also shown a biased result</ins>** which returned mostly only one class like below. Therefore,  **<ins>I am running another test (i3 using binary-class image dataset) to see if this issue only occurs when using multi-class dataset.</ins>** <br/><br/>
![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/3225e5ee-bc5d-4e15-9e51-6a4c0ea2634d) 

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

### s3: <ins>completed</ins>
  - running time
    > Slurm Job_id=240041 Name=s3.slurm Ended, Run time **01:25:21**
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/1e43b567-9bd4-46c4-908a-cc3030b8ffbe)
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/10#issue-1948916838 <br/><br/>

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

### t3: <ins>completed</ins> 
  - running time
    >  Slurm Job_id=240053 Name=t3.slurm Ended, Run time **01:11:30**
  - results
    >  ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/79021042-aeae-4ee2-8c5c-5eaffaac4c16) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/11#issue-1948967147 <br/><br/>

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
    > Slurm Job_id=209205 Name=i2.slurm Ended, Run time **2- 09:09:00** (I assume it means 2 days and 9h)
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/a7b69e2d-e3a3-4d06-9acd-d49271bc97e1) <br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/99a6acd0-eb6b-4dbd-ba71-551bfe76b16e) <br/>
  - best parameters
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/f94e58b6-ddff-4165-b56d-296698219539) <br/>
  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/9#issue-1943605147 <br/><br/>

### i3: <ins>running with data augmentation</ins>





