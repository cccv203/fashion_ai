## This repository is my record of attending the [Tianchi FashionAI competition](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100067.5678.1.43c15f59deBcQ1&raceId=231648)from March to June this year.It's just for my team and me to record ours thinking and attempts.
 
### At the begining:

*  *1:we have try five models for five category as we thought every category had a 
big gap from each others.*

*  *2:we have thought this is a single object keypoint detection as there is only one
thing in every piture.*

*  *3:we have thought if we want to win,we must have many GPUs and many samples,ps: this is
not wrong but some people can still beat us even under the same resources.*

****So, we have tried a best model in Single Person Pose Estimation,as it is [Learning Feature Pyramids for Human Pose Estimation](
https://arxiv.org/abs/1708.01101),which make sense for me as it's my first reimplement without a good code in github.
That time ,we got err 11.07,leaderboard 19th.****

### Then, we got some improvement, as it is:

*  *1:Thinking it is a Multi Person Pose Estimation question,so wo first got a object detection,like faster rcnn.*

*  *2:Reimplement the 1st model in MPPE,which is* [Cascaded Pyramid Network for Multi-Person Pose Estimation](https://arxiv.org/abs/1711.07319),
*At that time, we got it by ourselves as there is not open code.*

*  *3:There are some tricks like ,for loss design, wo got our OHKM and OHEM loss; for test ,wo got the mean of 11 largest points 
as our final point;for cpn bottleneck,we also using pyrnet bottleneck.*


<div align="center">
<img src="data/imgs/demo/0013.png" width="256" hidth="256"/>
<img src="data/imgs/demo/0014.png" width="256" hidth="256"/>
<br>
<img src="data/imgs/demo/0015.png" width="256" hidth="256"/>
<img src="data/imgs/demo/0016.png" width="256" hidth="256"/>
</div>

##Extending,At final round,we got some inspires, as :

*  *1:Another loss ,which we think can improve a lot as it solve the disadvantage in heatmap argmax.*
<div align="center">
<img src="data/imgs/demo/0001.png" width="256" hidth="256"/>
<img src="data/imgs/demo/0002.png" width="256" hidth="256"/>
</div>

*  *2:Using dilation conv to keep resolution and expand receptive field,and change dilation rate to 
react gridding problem.*

<div align="center">
<img src="data/imgs/demo/0003.png" width="256" hidth="256"/>
<img src="data/imgs/demo/0004.png" width="256" hidth="256"/>
</div>

*  *3:Another team show a Multi-level Unit to fight against Resnet Unit,sound interesting but we can not get finetune model using new unit.*

<div align="center">
<img src="data/imgs/demo/0005.png" width="256" hidth="256"/>
<img src="data/imgs/demo/0006.png" width="256" hidth="256"/>
</div>

*  *4:The 1st team, show a new model named SHN.*
<div align="center">
<img src="data/imgs/demo/0007.png" width="256" hidth="256"/>
<img src="data/imgs/demo/0008.png" width="256" hidth="256"/>
<br>
<img src="data/imgs/demo/0009.png" width="256" hidth="256"/>
<img src="data/imgs/demo/0010.png" width="256" hidth="256"/>
<p>Network Design</p>
<img src="data/imgs/demo/0011.png" width="256" hidth="256"/>
<img src="data/imgs/demo/0012.png" width="256" hidth="256"/>
<p>Training Process</p>
</div>

### In the future, we will implement these one by one.