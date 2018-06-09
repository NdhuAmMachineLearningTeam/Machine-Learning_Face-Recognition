# 說明文件
<p>首先要先導入以下這些Packages：</p>
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
不過這次Face database每個人的相片都是15張且命名有條序，所以用for迴圈控制名稱來讀入也OK，不一定要使用。<p>
更正 - Face database是已經從15張裡面抽掉了2張作測試資料，而抽掉的編號是隨機的，所以用os讀入為佳。
</li>
<li>
<strong>cv2 </strong>，即OpenCV2.0，可以幫助我們對圖像進行一些幾何變換(Geometric Transformation)，而等等要使用到的是伸縮的部分(Scaling)，
interpolation參數的使用可參考 - <a href="http://monkeycoding.com/?p=609">影像尺寸改變(resize)</a>
</li>
</ol>

<pre><code>import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
</pre></code>

<ul>
<li>
首先要注意到的是每張照片的尺寸(像素)並不相同
</li>
<li>
再來要注意的是每張圖的長寬比也不一樣
</li>
</ul>
<p>這也是為甚麼一開始我試了好久都沒能把全部data存成一個4維的ndarray，但對單張做卻可以正常轉換。</p>
<p>所以想說使用opencv套件將圖片都換成最大或最小的size，也多虧如此才注意到換成最大size時圖片變形了。<p>
<p>選最大或最小是考量到interpolation參數的適性，想盡量統一為拉長或縮小，
而最小size雖然維度比較低，但從結果可以看到圖像明顯變模糊，而放大的則與原圖相近，故先考慮以放大作為我們resize的方向。</p>
<p>再來考慮到長寬比的問題，將所有圖的長寬比取期望值試著最小化對每張圖的影響，並且把圖片都放大為一個符合該比例的size。</p>
<p>不過嚴謹一點的話可能還要確認一下比例的分配(distribution)，看是否有會對mean造成嚴重影響的極端值，或整體是否為多峰分配(multi-modal)，即資料可分為2種或多種不同的特定比例等等。</p>
<p>在對inputdata進行上述的處理後我們即可把其變成一個4維的ndarray形式，其中第一個維度為我們圖片的資料數，2,3維是其長與寬，第4維則是彩圖每個像素點的RGB色光三原色。</p>
<p>統一data的size是我目前預想的進行方向，不一定是最好的方法，也很有可能不用把size統一就能訓練，不過這樣W權重矩陣的運算會有很多奇怪的狀況，覺得乍看之下沒甚麼搞頭，當然，也可能有更好的size處理方式，有想到的話我們到時候都可以再去改。</p>
<p>還有一個需要注意的是每個訓練點都還有RGB三個分開的值，是否能就這樣訓練還是需要做特別的處理在進行到訓練模型開發階段時可能也要再研究下。</p>
<p>　</p>
