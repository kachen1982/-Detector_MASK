# -口罩偵測示警功能 應用於Raspi 4.
綜觀:
COVID-19的影響下，近年的口罩政策，無法快速地解除戴口罩的現象。
導致因未正確配戴口罩，經糾正後發生的攸關生命安全的社會事件發生，如在商業空間中，有偵測口罩的設備提醒違規者，可避免掉不必要的寶貴生命失去。
這是一個基礎型的口罩偵測端點系統，透過WEBCAM設備進行人臉判斷是否有戴口罩警示的功能，初期以edge 方式呈現功能，後續還可以加入上傳雲端做相關分析動作。

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/_uxsrwm8nIA/0.jpg)](http://www.youtube.com/watch?v=_uxsrwm8nIA)


# -使用相關軟硬體設備
Hardward

Raspberry Pi 4 *1

Sony Webcam *1

Jumper wires

breadboard *1

LED red *1 , green *1

power adapter / cable *1

Software

Python 3.7

Tensorflow 2.1

OPEN CV

imutils

# -Raspberry Pi 面罩檢測器項目如何工作？

當用戶接近您的網絡攝像頭時，利用 TensorFlow、OpenCV 和 imutils 包的 Python 代碼將檢測用戶是否佩戴口罩。 
戴口罩的人員將在其臉部周圍看到一個綠色框，上面寫著“Thank you. Mask On.” 
不戴口罩的人員會在他們的臉上看到一個紅色框，上面寫著“No Face Mask Detected.”

# -樹莓派面罩檢測器項目是如何進行的？
  # -Part 1 
  
