首先要先導入以下這些Packages：<p>
<ol>
<li>
<strong>numpy </strong>用來處理多維矩陣的運算<p>
<p>可以在這個網站看有哪些指令能使用和其傳入參數的方式，大部分的操作應該都不用自己寫了- <a href="https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html">The N-dimensional array</a></p>
</li>
<li>
<strong>matplotlib:</strong>分別幫助我們以ndarray的形式讀入圖片和做出繪圖，使用上類似matlab。<p>
</li>
<li>
<strong>os </strong>在這邊是被我用來讀取檔案，<code>os.listdir</code>可以直接抓到一個資料夾內所有的檔案名稱<p>
不過這次Face database每個人的相片都是15張且命名有條序，所以用for迴圈控制名稱來讀入也OK，不一定要使用。
</li>
<li>
<strong>cv2 </strong>，即OpenCV2.0，可以幫助我們對圖像進行一些幾何變換(Geometric Transformation)，而等等要使用到的是伸縮的部分(Scaling)，
interpolation參數的使用可參考 - <a href="http://monkeycoding.com/?p=609">影像尺寸改變(resize)</a>
</li>

<pre><code>import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
</pre></code>
