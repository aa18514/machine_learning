## Troubleshooting issues

At this point in time, there are some issues with the Windows version of
Auto-Keras given that you use 'CUDA' device instead of 'CPU'. It was 
experienced that the multiprocessing API and PyTorch do not work well together. The issues are listed as follows: <br>

```terminal
Traceback (most recent call last):
<br>... <br>
accuracy, loss, graph = train_results.get()[0] <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\
\lib\multiprocessing\pool.py", line 644, in get
raise self._value <br>
TypeError: 'float' object cannot be interpreted as an integer". <br>
```
Don't be mislead into thinking that the problem is related
to the multiprocessing API, it simply refers to explicitly casting values in the tuple
'self.padding' as int (C:\Users\user\AppData\Local\Programs\Python\Python36\site-packages\torch\nn\modules\conv.py at line 301 before calling the function F.conv2d with appropriate parameters)

* After fixing the former error, when you run the 'auto_keras_magic.py' script again, the program will fail with the following information

```terminal
Traceback (most recent call last):
<br>... <br>
THCudaCheck FAIL file=c:\new-builder_3\win-wheel\pytorch\torch\csrc\generic\StorageSharing.cpp line 231 error 71: operation not supported 
<br> ... <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\lib\site-packages\autokeras\search.py", line 190 in search <br>
accuracy, loss, graph = train_results.get()[0] <br>
File "C:\Users\user\AppData\Local\Programs\Python\Python36\
\lib\multiprocessing\pool.py", line 644, in get
raise self._value <br>
multiprocessing.pool.MaybeEncodingError: error sending result '[(98.08, tensor=(2.3784, device='cuda:0'), <autokeras.graph.Graph.object at 0x000002821B58E668>)]' <br>
Reason: 'RuntimeError('cuda runtime error (71) : operation not supported at c:\\new-builder_3\\win-wheel\\pytorch\\torch\\csrc\\generic\\StorageSharing.cpp:231',)' <br>
```

Unfortunately it looks like multiprocessing and PyTorch do not seem to work well together. A hack to this problem is to replace line 178: <br>
"train_results = pool.map_async(train, [(graph, train_data, test_data, self.trainer_args, <br>
                                                os.path.join(self.path, str(model_id) + '.png'), self.verbose)])" <br>
with: <br>
train_results = train((graph, train_data, test_data, self.trainer_args, os.path.join(self.path, str(model_id) + '.png'), self.verbose)) <br>
and replace line 190 <br>
accuracy, loss, graph = train_results.get()[0] <br>
with: <br>
accuracy, loss, graph = train_results <br>

* Another issue is when if you don't call the time-limit the GPU may run out of memory, if you keep the time-limit parameter to a reasonably small value such as 10 seonds, you will exhaust your GPU memory resources.
Unfortunately, this is a bug, you could get around this by specifying the time_limit to a reasonably small value such as 10 seconds to ensure the run_searcher_once method runs only once.

if you check the following:

```terminal
user@ubuntu:~$ vim C:\Users\user\AppData\Local\Programs\Python36\lib\site-packages\autokeras\classifier.py
``` 
you will find the following piece of code on line 223

```terminal

if time_limit is None:
    time_limit = 24 * 60 * 60

start_time = time.time()
while time.time() - start_time <= time_limit:
    run_searcher_once(train_data, test_data, self.path)
    if len(self.load_searcher().history) >= Constant.MAX_MODEL_NUM:
        break

```

you could see the flaw in this piece of code <br>
if time_limit parameter is not specified it defaults to 24 * 60 minutes <br>
the default value of Constant.MAX_MODEL_NUM is 1000 <br>
so you keep on looping in the while loop until len(self.load_searcher().history) >= Constant.MAX_MODEL_NUM <br>
also after the train process is complete self.load_searcher().history stores the new trained model which means its length only increases by one <br>
you could get around this by maybe replacing Constant.MAX_MODEL_NUM to a sane value like 1 (or choose the time limit to be low like 10 seonds), I hope this helps....
