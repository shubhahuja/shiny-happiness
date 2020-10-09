# **Stone Paper Scissors**

**We all have played stone paper scissors or maybe some variation of it, my favorite variation is Stone-paper-scissors-lizard-spork and that&#39;s possibly gonna be the updated version of this game but for now, we have the classic!**

**Background**

Well, the idea came when I was learning convolutions. My 1st project was differentiating dogs and cats, I know classic! but then I wanted to experiment more so here I am

**How it Works**

So I made everything from scratch I could have used VGG or some other model but kind of wanted to make a whole CNN

My model has 4 convolution layers and some dropouts, and also did some augmentation to my dataset to get a better result

The project has an opencv.py file which on running asks for some images you want to take for each gesture

Oh! did I not mention you don&#39;t need to download a big data set!

Although the script takes any number of images you want to take however for good predictions it better to have more than 200 images of each type the images are of 4 types rock, paper, scissor, and background(none)

All you have to do is launch the script, pass a number, place your hand in a grid with the asked gesture and press a!

The computer does the rest for you

Then we do the training

Run tensorflow.py

might take a few seconds and you are all set!

Run play.py and ENJOY!
