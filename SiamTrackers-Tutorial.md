# VSCode
## 1 Install 

    Download ubuntu version from   https://code.visualstudio.com/Download
    
    sudo dpkg -i  xxx.deb  


## 2 Shortcuts

    Show Command: Ctrl+Shift+P
    Open Files:   Ctrl+O
    Open Folder: Ctrl+K, Ctrl+O

## 3 Extensions

    Install Python 
    Install Code Runner
    Install Rainbow Brackets 
    Install Markdown preview enhanced



## 5 Select  Indentation

## 6  Font Size

    Bigger: Ctrl+ '+'  
    Smaller: Ctrl+ '-'  

## 7 Theme Style

	files-->preferences-->color theme
	https://www.cnblogs.com/qianguyihao/archive/2019/04/18/10732375.html

## 8 Local Debugging
    Select Python interpreter'virtual environment
    Debug: F5 
    Step over: F10 
    Step into: F11 
    Run Coder: Alt+Ctrl+N

## 9 Configuration SSH

    Linux:
    sudo apt-get install openssh-server
    sudo apt-get install openssh-client
    
    cd /home/username/
    ssh-keygen -t rsa -P ""
    cd ~/.ssh 
    ssh-copy-id   reomte-host-name@romote-host-ip
    
    Note: windows system can not find 'ssh-copy-id' command, you should open the xxx.pub file (/C/xxx/xxx/.ssh/xxx.pub) of your windows computer, then copy the **contents** of the file to the  authorized_keys file of the remote host manually. If your remote host does not have authorized_keys, you should manually create it before performing the copy operation:
        cd ~/.ssh (Remote host)
        touch authorized_keys
        vim  authorized_keys
    
    Configure the name of the remote host to relize password-free login
    cd ~/.ssh 
    touch config
    vim config 
    
    Then add tehe following content to the config file:
    Host server1(The nick name of the remote host)    
        HostName ip address(The ip address of remote host)    
        User root (The user name of remote host)    
        port 22 (The port of remote host)


​    
## 10  Remote Debugging
	Install Remote development
	Display Image:
	Install Remote x11 (https://blog.csdn.net/zb12138/article/details/107160825/)

# Parallel Train

## 1 Single GPU
	1 Set the visible GPU number
	
	Method 1
	CUDA_VISIBLE_DEVICES=0,1  python xxx.py 
	
	Method 2
	vim xxx.py 
	os.environ['CUDA_VISIBLE_DEVICES']='0,1'
		
	2 Put the model, data, and loss function on the GPU device 
	
	Method 1
	model=model.cuda() 
	Loss=Loss.cuda()
	
	Method 2 
	device=torch.device('cuda:{}'.format(args.gpu_id))
	model=model.to(device)

## 2 DataParallel (Multi GPU)

	model = torch.nn.DataParallel(model.cuda())


## 3 DistributedDataParallel  (Multi GPU)

	1 Run in terminal
	
	python -m torch.distributed.launch --nproc_per_node=NUM_GPUS  ./bin/dist_train.py 
	
	2 Parse parameters
	
	parser.add_argument("--local_rank", type=int,default=0)
	
	    2 nodes, 2 GPUs for per node
	    rank=0,1,2,3 
	    node1: local_rank= 0 or 1 (rank%node_num)
	    mode2: local_rank= 0 or 1 
		
	3 Initialize communication method		  
	
		torch.distributed.init_process_group(backend='nccl', init_method='env://')
	
	4 Set the GPU number that the current process needs to use
		torch.cuda.set_device(args.local_rank)
		
	5 Generate  corresponding data labels for each  proceess
		
		train_sampler=torch.utils.data.distributed.DistributedSampler(train_data)
		train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=F
	alse,num_workers=2, pin_memory=True, sampler=train_sampler)
	
	Before each epoch, the shuffle effect is achieved by calling the following commands:
		train_sampler.set_epoch(epoch) 
		
	6 Calculate the  loss, summarize the information of each process
	
		Example:	
	
	    def reduce_tesnor(tensor):
	        # sum the tensor data across all machines
	        dist.all_reduce(rt. op=dist.reduce_op.SUM)
	        return rt
	
	     output=model(input)
	
	     loss=Loss(output,label)
	
	     log_loss = reduce_tensor(loss.clone().detach_())
	
	     torch.cuda.synchronize()  # wait every process finish above transmission
	
	     loss_total += log_loss.item()   
	     
	7 Avoid conflicts when writing log files or print
		if args.local_rank==0:
			print('xxxx')
			log.info('xxxx')


# SiamFC

## 1 SiamFC(VID)

	1 Train 
		python ./bin/my_train.py
	
	2 Test and Evaluate  with got10k-toolkit
		python ./bin/my_test.py
	
	3 Evaluate
		python ./bin/my_eval.py

## 2 SiamFC-GOT

	1 Train 
		python ./bin/my_train.py
	
	2 Test and Evaluate with got10k-toolkit
		python ./bin/my_test.py
	
	3 Batch Test   
		./bin/cmd_test.sh
	
	4 Evaluate 
		python ./bin/my_eval.py
	
	5 Hyperparameter
		python ./bin/hp_search.py

# SiamRPN

## 1 SiamRPN (YTB&VID)

	1 Generate training set
		python ./bin/create_dataset_ytbid.py
	
	2 Generate Lmdb file
		python ./bin/create_lmdb.py
	
	3 Train
		python ./bin/my_train.py
	
	4 Test and Evaluate
		python ./bin/my_test.py

## 2 SiamRPN-GOT

	1 Train
		python ./bin/my_train.py
	
	2 Test and  Evaluate
		python ./bin/my_test.py
	
	3 Batch Test
		./bin/cmd_test.sh
	
	4 Hyperparameter
		python ./bin/hp_search.py
	
	5 DDP Train 
		./bin/cmd_dist_train.sh

## 3 SiamRPNpp-UP
	Note that you should first build region by run the follow command: 
		python setup.py build_ext —-inplace
		
	1 Train
		python ./bin/my_train.py
	
	2 Test 
		python ./bin/my_test.py
	
	3 Batch Test
		./bin/cmd_test.sh
	
	4 Batch Evaluate
		./bin/cmd_eval.sh
	
	5 Demo
		python ./bin/my_demo.py
	
	6 Hyperparameter
		python ./bin/hp_search.py
	
	7 DDP Train 
		./bin/cmd_dist_train.sh

# DaSiamRPN

## 1 DaSiamRPN
	1 Test 
		python ./bin/my_test.py

## 2 DaSiamRPN-GOT

	1 Train
		python ./bin/my_train.py
	
	2 Test 
		python ./bin/my_test.py
	
	3 Batch Test
		./bin/cmd_test.sh
	
	4 Hyperparameter
		python ./bin/hp_search.py

## 3 SiamRPNpp-DA

	Note that you should first build region by run the follow command: 
	python setup.py build_ext —-inplace
	
	1 Train
		python ./bin/my_train.py
	
	2 Test 
		python ./bin/my_test.py
	
	3 Batch Test
		./bin/cmd_test.sh
	
	4 Batch Evaluate
		./bin/cmd_eval.sh
	
	5 Demo
		python ./bin/my_demo.py
	
	6 Hyperparameter
		python ./bin/hp_search.py
	
	7 DDP Train 
		./bin/cmd_dist_train.sh

# UpdateNet

## 1 UpdateNet-FC


	1 Generate training set
		python ./updatenet/create_template.py
	
	2 Train UpdateNet (Note thae you should change the stage value )
		python ./updatenet/train_upd.py
	
	3 Test UpdateNet (Note that you should set udpatenet path and stage value)
		python ./bin/my_test.py 

## 2 UpdateNet-DA


    1 Generate training set
    	python ./updatenet/create_template.py
    
    2 Train UpdateNet (Note that you should change the stage value 
    )
    	python ./updatenet/train_upd.py
    
    3 Test UpdateNet (Note that you should set udpatenet path and stage value)
    	python ./bin/my_test.py 


   	


## 3 UpdateNet-UP
	1 Generate training set
		python ./updatenet/create_template.py
		
	2 Train UpdateNet
		python ./updatenet/train_upd.py
	
	3 Test UpdateNet
		python ./bin/my_test.py 

## 4 UpdateNet-DW 

	Note that you should first build region by run the follow command: 
	python setup.py build_ext —-inplace
	
	1 Generate training set
		python ./updatenet/create_template.py
	
	2 Train UpdateNet
	Note that you should change the stage value 
		python ./updatenet/train_upd.py 
	
	3 Test UpdateNet (Note that you should set udpatenet path and stage value)
		python ./bin/my_test.py 

# SiamDW 

## 1 SiamDW-FC
	1 Train
		python ./bin/my_train.py
	
	2 Teat and Evaluate
		python ./bin/my_test.py
	
	3 Batch Test
		python ./bin/cmd_test.py
	
	3 Hyperparameters
		python ./bin/hp_search.py

## 2 SiamDW-UP
	1 Train
		python ./bin/my_train.py
	
	2 Teat and Evaluate
		python ./bin/my_test.py
	
	3 Batch Test
		python ./bin/cmd_test.py
	
	3 Hyperparameters
		python ./bin/hp_search.py

# SiamRPNpp

## 1 SiamRPNpp-DW

	Note that you should first build region by run the follow command: 
	python setup.py build_ext —-inplace
	
	1 Train
		python ./bin/my_train.py
	
	2 Test 
		python ./bin/my_test.py
	
	3 Batch Test
		./bin/cmd_test.sh
	
	4 Batch Evaluate
		./bin/cmd_eval.sh
	
	5 Demo
		python ./bin/my_demo.py
	
	6 Hyperparameter
		python ./bin/hp_search.py
	
	7 DDP Train 
		./bin/cmd_dist_train.sh

##  2  SiamRPNpp-ResNet

# SiamFCpp

## 1 SiamFCpp-GOT

	1 first, you should run compile.sh 
		sh ./compile.sh
	
	2 Train
	
		python ./bin/my_train.py
	
	3 Test
	
		python ./bin/my_test.py
	
	Experiments
	
	    OTB 2015 
	    "success_score": 0.6289266117015362,
	    "precision_score": 0.830571693318284,
	    "success_rate": 0.7891486658050533,
	    "speed_fps": 84.38537836344958
	
	    official
	    "success_score": 0.6797143434600249,  
	    "precision_score": 0.8841645010368359,
	    "success_rate": 0.8551268591684209,
	    "speed_fps": 144.9084986738754

## 2 SiamFCpp-GoogleNet


# Experiments

|   Trackers|       | SiamFC   | SiamRPN | SiamRPN | DaSiamRPN |DaSiamRPN | SiamRPNpp | SiamRPNpp | SiamRPNpp | SiamRPNpp | SiamFCpp |SiamFCpp |
|:------------:|:-----:|:--------:   | :------:    |:------:  |:------: |:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Train Set |       | GOT | official | GOT | official | official | official | GOT | GOT | GOT | GOT | official |
|  Backbone |  | Group | AlexNet | AlexNet |  AlexNet  |    DA     | DW    | DW  | UP | DA | AlexNet |AlexNet|
|     FPS   |     |   85   |   >120   |   >120   |   >120        |   >120 |    >120      |    >120  |    >120  | >120  | >120 |    >120 |
|           |       |           |           |           |            |         |         |         |         |         |         |        |
| OTB100    |  AUC   |  0.589  | 0.637 | 0.603 |   0.655   |  0.646   |   0.648  |  0.623  |  0.619  |  0.634  |  0.629  | **0.680**    |
|           |  DP   |   0.794   | 0.851 | 0.820 |   0.880   |  0.859   |  0.853   |  0.837  |  0.823  |  0.846  |  0.830  | **0.884**   |
|           |     |           |           |           |            |         |         |         |         |         |         |        |
| UAV123    |  AUC  |   0.504    | 0.527 |  |   0.586   |  0.604   |  0.578   |     |     |     |     |  **0.623**    |
|           |  DP   |    0.702   | 0.748   |    |   0.796   | **0.801**    |  0.769   |     |     |     |     |  0.781   |
|           |     |           |           |           |            |         |         |         |         |         |         |        |
| UAV20L    |  AUC  |  0.410     | 0.454 |  |         |   0.524  |  **0.530**   |     |     |     |     |  0.516  |
|           |  DP   |   0.566    | 0.617 |  |         | **0.691**   |  0.695   |     |     |     |     |  0.613   |
|           |     |           |           |           |            |         |         |         |         |         |         |        |
| DTB70     |  AUC  |    0.487   |       |       |          |  0.554|   0.588  |     |     |     |     | **0.639**   |
|           |  DP   |    0.735   |       |       |         |   0.766|   0.797  |     |     |     |     |  **0.826**   |
|           |       |           |           |           |            |         |         |         |         |         |         |        |
| UAVDT     |  AUC  |   0.451 |    |    |           |  0.593  |  0.566   |     |     |     |     |  **0.632**    |
|           | DP    |   0.710 |    |    |           |  0.836  |  0.793   |     |     |     |     |   **0.846**   |
|           |     |           |           |           |            |         |         |         |         |         |         |        |
| VisDrone-Train  | AUC   |    0.510|    |    |           |   0.547 |  0.572   |     |     |     |     |  **0.588**    |
|           |  DP   |    0.698|    |    |           |   0.722 |   0.764  |     |     |     |     |  **0.784**    |
|           |     |           |           |           |            |         |         |         |         |         |         |        |
| VOT2016   |  A  |   0.538    |  0.56 |   |  0.61      |  0.625   |  0.618   | 0.582 |     |     | 0.612 |  **0.626**    |
|           | R     |    0.424   | 0.26   |    |  0.22      |  0.224   |  0.238   | 0.266 |     |     | 0.266 |   **0.144**   |
|           | E     |    0.262   | 0.344   |    |  0.411     |  0.439   |  0.393   | 0.372 |     |     | 0.357 |  **0.460**    |
|           |Lost   |    91      |          |          |           |  48      |  51      | 57 |        |        | 57 |    31  |
|           |     |           |           |           |            |         |         |         |         |         |         |        |
| VOT2018   | A     |     0.501  | 0.49  |   |   0.56     |  **0.586**   | 0.576    | 0.563 | 0.555/0.562 | 0.557 | 0.584 | 0.577   |
|           |  R    |    0.534   | 0.46   |    |   0.34     |  0.276   |  0.290   |  0.375  |  0.384/0.398  |  0.412  | 0.342 | **0.183**   |
|           | E     |    0.223   | 0.244   |    |   0.326    | 0.383    |  0.352   |  0.300  |  0.292/0.292  |  0.275  | 0.308 | **0.385**   |
|           | Lost  |   114      |         |         |           |  59      |   62       |   80     |   82/85   |   88   | 73 |   39     |




| Year | Conf |               Trackers               | Debug | Train | Test |      |      | Data | Set  |      |       |   Toolkit    |   Source   |
| :--- | :--- | :----------------------------------: | ----- | :---: | :--: | :--: | :--: | :--: | :--: | :--: | :---: | :----------: | :--------: |
|      |      |                                      |       |       |      | VID  | DET  | COCO | YTB  | GOT  | LaSOT |              |            |
| 2016 | ECCV |     SiamFC      | ✔     |   ✔   |  ✔   |  ✔   |      |      |      |      |       |    got10k    | unofficial |
|      |      | SiamFC      | ✔     |   ✔   |  ✔   |      |      |      |      |  ✔   |       |    got10k    | unofficial |
|      |      |                                      |       |       |      |      |      |      |      |      |       |              |            |
| 2018 | CVPR |    SiamRPN   | ✔     |   ✔   |  ✔   |  ✔   |      |      |  ✔   |      |       |    got10k    | unofficial |
|      |      | SiamRPN | ✔     |   ✔   |  ✔   |      |      |      |      |  ✔   |       |    got10k    | unofficial |
|      |      |                                      |       |       |      |      |      |      |      |      |       |              |            |
| 2018 | ECCV |      DaSiamRPN     | ✔     |       |  ✔   | ✔ | ✔ | ✔ | ✔ |      |       |    pysot     | official |
|      |      |      DaSiamRPN  |   ✔   |   ✔    |   ✔   |      |      |      |      |  ✔    |       |  pysot | unofficial   |
|      |      |                                      |       |       |      |      |      |      |      |      |       |              |            |
| 2019 | ICCV |     UpdateNet(FC)     | ✔     |   ✔   |  ✔   |      |      |      |      |      |   ✔   |    pysot     | unofficial |
|  |  | UpdateNet(UP) | ✔ | ✔ | ✔ | | | | | | ✔ | pysot | unofficial |
|  |  | UpdateNet(DA) | ✔ | ✔ | ✔ | | | | | | ✔ | pysot | official |
|  |  | UpdateNet(DW) | ✔ | ✔ | ✔ | | | | | | ✔ | pysot | unofficial |
|      |      |                                      |       |       |      |      |      |      |      |      |       |              |            |
| 2019 | CVPR |       SiamDW(FC)       | ✔     |   ✔   |  ✔   |      |      |      |      |      |   ✔   | got10k |  unofficial  |
|  |  | SiamDW(UP) | ✔ | ✔ | ✔ |  |  |  |  |  | ✔ | got10k | unofficial |
|      |      |                                      |       |       |      |      |      |      |      |      |       |              |            |
| 2019 | CVPR | SiamRPNpp(DW) | ✔     |   ✔   |  ✔   |  ✔   |  ✔   |  ✔   |  ✔   |      |       |    pysot     | official |
|      |      |SiamRPNpp(DW)| ✔     |   ✔   |  ✔   |      |      |      |      |  ✔   |       |    pysot     |  unofficial|
| | |SiamRPNpp(UP)| ✔ | ✔ | ✔ | | | | | ✔ | | pysot | unofficial |
| | |SiamRPNpp(DA)| ✔ | ✔ | ✔ | | | | | ✔ | | pysot | unofficial |
| | |SiamRPNpp(ResNet)| ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ |  | | pysot | official |
|      |      |                                      |       |       |      |      |      |      |      |      |       |              |            |
| 2019 | CVPR |       SiamMask      | ✔     |   ✔   |  ✔   |  ✔   |  ✔   |  ✔   |  ✔   |      |       |    pysot     |  official  |
|      |      |                                      |       |       |      |      |      |      |      |      |       |              |            |
| 2020 | AAAI |  SiamFCpp | ✔     |   ✔   |  ✔   |  ✔   |  ✔   |  ✔   |  ✔   |  ✔   |   ✔   | pysot&got10k | official |
|      |      | SiamFCpp | ✔     |   ✔   |  ✔   |      |      |      |      |  ✔   |       | pysot&got10k | unofficial |
| | | SiamFCpp(GoogleNet) | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | pysot&got10k | official |

