# 041023
Jin) 
- The s1, s2, and s3 codes have been modified and completed so far. (other codes have not been changed since the dissertation submission.) <br/>
- Even though the s3 model's code has been changed ("max_trials=1", which might have occurred the low performance, was set in the previous s3 model and now deleted), <ins>the results, especially accuracy, are quite the same.</ins> However, <ins>the best "number of hidden layers" hyperparameter value has been changed from 5 to 3</ins> (*total number of dense layers including an input layer). Please refer to the s2 result. <br/><br/>
  *The existing result and best hyperparameters<br/>
  > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/5f88408a-13fc-4942-9130-14bc750f3313) <br/><br/>
<br/>

## Structured data models
### s1: Completed
  - running time
    > Slurm Job_id=194887 Name=s1.slurm Ended, Run time **00:02:04**

  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/83931863-bd44-437d-8417-6159538f9b74) <br/>

  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/1#issue-1920581046 <br/><br/>


### s2: Completed (kerastuner 1.3.5)
  - running time
    > Slurm Job_id=193664 Name=s2.slurm Ended, Run time **06:18:18**
    
  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/68dc9680-170c-40e5-b20a-7e65e2f59acb) <br/>
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/64ff103b-50f1-4ef0-a613-245724bfd4aa) <br/>

  - best hyperparameters
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/ea6b92b5-4b98-4c76-960f-b2c460ae94ba) <br/>

   - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/2#issue-1925415370 <br/><br/>


### s3: Completed (autokeras 1.0.16)
  - running time
    > Slurm Job_id=193796 Name=s3.slurm Ended, Run time **01:26:29**

  - results
    > ![image](https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/assets/135728064/cb4bfe55-9e95-43c6-b11b-a766a1d8b03f) <br/>

  - The full result can be confirmed here https://github.com/youngmin-jin/python-multi-class-classification-keras-autokeras-comparison/issues/3#issue-1925415683 <br/><br/>




