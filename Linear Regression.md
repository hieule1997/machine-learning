 # Linear Regression   
Đây là một thuật toán **Supervised learning** có tên **Linear Regression** (Hồi Quy Tuyến Tính).   
Bài toán được đưa ra :   
 - Nếu một căn nhà rộng x1 mét vuông, có x2 phòng ngủ và cách trung tâm thành phố x3 km có giá là bao nhiêu. Giả sử chúng ta đã có số liệu thống kê từ 1000 căn nhà trong thành phố đó, liệu rằng khi có một căn nhà mới với các thông số về diện tích, số phòng ngủ và khoảng cách tới trung tâm, chúng ta có thể dự đoán được giá của căn nhà đó không? 

**Chú ý 1:** 
y là giá trị thực của outcome (dựa trên số liệu thống kê chúng ta có trong tập training data), trong khi ^y là giá trị mà mô hình Linear Regression dự đoán được. Nhìn chung, y và ^y là hai giá trị khác nhau do có sai số mô hình, tuy nhiên, chúng ta mong muốn rằng sự khác nhau này rất nhỏ.

**Chú ý 2**: Linear hay tuyến tính hiểu một cách đơn giản là thẳng, phẳng. Trong không gian hai chiều, một hàm số được gọi là tuyến tính nếu đồ thị của nó có dạng một đường thẳng. Trong không gian ba chiều, một hàm số được goi là tuyến tính nếu đồ thị của nó có dạng một mặt phẳng. Trong không gian nhiều hơn 3 chiều, khái niệm mặt phẳng không còn phù hợp nữa, thay vào đó, một khái niệm khác ra đời được gọi là siêu mặt phẳng (hyperplane). Các hàm số tuyến tính là các hàm đơn giản nhất, vì chúng thuận tiện trong việc hình dung và tính toán. Chúng ta sẽ được thấy trong các bài viết sau, tuyến tính rất quan trọng và hữu ích trong các bài toán Machine Learning. Kinh nghiệm cá nhân tôi cho thấy, trước khi hiểu được các thuật toán phi tuyến (non-linear, không phẳng), chúng ta cần nắm vững các kỹ thuật cho các mô hình tuyến tính

## Phân tích toán học
### Dạng của Linear Regression :
 ![anh](/Image/bieuthucLinearRegression.png)

### Sai số dự đoán :
![anh](/Image/saisoLineaRegression.png)
### Hàm mất mát :
![anh](/Image/hammatmatLinearRegression.png)
### Nghiệm cho bài toán Linear Regression
Cách phổ biến nhất để tìm nghiệm cho một bài toán tối ưu (chúng ta đã biết từ khi học cấp 3) là giải phương trình đạo hàm (gradient) bằng 0! Tất nhiên đó là khi việc tính đạo hàm và việc giải phương trình đạo hàm bằng 0 không quá phức tạp. Thật may mắn, với các mô hình tuyến tính, hai việc này là khả thi.  
![anh](/Image/daohammatmat.png)

Các bạn có thể tham khảo bảng đạo hàm theo vector hoặc ma trận của một hàm số trong mục [D.2 của tài liệu này.](https://ccrma.stanford.edu/~dattorro/matrixcalc.pdf)     

Phương trình đạo hàm bằng 0 tương đương với:
![anh](/Image/ph.png)  
Với khái niệm giả nghịch đảo, điểm tối ưu của bài toán Linear Regression có dạng:  
![anh](/Image/gianghichdao.png) 
## Ví dụ trên Python
Trong phần này, tôi sẽ chọn một ví dụ đơn giản về việc giải bài toán Linear Regression trong Python. Tôi cũng sẽ so sánh nghiệm của bài toán khi giải theo phương trình giả nghich đảo và nghiệm tìm được khi dùng thư viện [scikit-learn](https://scikit-learn.org/stable/) của Python. (Đây là thư viện Machine Learning được sử dụng rộng rãi trong Python). Trong ví dụ này, dữ liệu đầu vào chỉ có 1 giá trị (1 chiều) để thuận tiện cho việc minh hoạ trong mặt phẳng.  

Chúng ta có 1 bảng dữ liệu về chiều cao và cân nặng của 15 người như dưới đây: 
 
|Chiều cao (cm)	|Cân nặng (kg)	|Chiều cao (cm)	|Cân nặng (kg)|

|147	|49	|168	|60|
|150	|50	|170	|72|
|153	|51	|173	|63|
|155	|52	|175	|64|
|158	|54	|178	|66|
|160	|56	|180	|67|
|163	|58	|183	|68|
|165	|59	 	

| Chiều cao (cm)  |     Cân nặng (kg)     |  Chiều cao (cm) |Cân nặng (kg)|
|----------|:-------------:|------:|--------:|
|147	|49	|168	|60|
|150	|50	|170	|72|
|153	|51	|173	|63|
|155	|52	|175	|64|
|158	|54	|178	|66|
|160	|56	|180	|67|
|163	|58	|183	|68|
|165	|59	 	
  
Tùy vào từng bài toán ta xác định có nên xử dụng Linear Regression. Nhưng trong bài toán này ta nhận thấy rằng cân nặng sẽ tỉ lệ thuận với chiều cao 

### Hiển thị dữ liệu trên đồ thị   
Trước tiên, chúng ta cần có hai thư viện [numpy](http://www.numpy.org/) cho đại số tuyến tính và [matplotlib](https://matplotlib.org/) cho việc vẽ hình.

```
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
```

Tiếp theo, chúng ta khai báo và biểu diễn dữ liệu trên một đồ thị.Sử dụng numpy để khai báo các mảng dữ liệu đầu vào 
```
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data 
plt.plot(X, y, 'ro')
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```

![anh](/Image/plt.png)


  Từ đồ thị này ta thấy rằng dữ liệu được sắp xếp gần như theo 1 đường thẳng, vậy mô hình Linear Regression nhiều khả năng sẽ cho kết quả tốt:  

`(cân nặng) = w_1*(chiều cao) + w_0`

### Nghiệm theo công thức
Chúng ta sẽ tính toán các hệ số w_1 và w_0 dựa vào công thức điểm tối ưu bài toán  
 **Chú ý**: giả nghịch đảo của một ma trận A trong Python sẽ được tính bằng numpy.linalg.pinv(A) . pinv là từ viết tắt của pseudo inverse.
```
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
```
  
Kết quả nhận được : 
```
w =  [[-33.73541021]
      [  0.55920496]]
```
![anh](/Image/pltline.png) 
  Từ đồ thị bên trên ta thấy rằng các điểm dữ liệu màu đỏ nằm khá gần đường thẳng dự đoán màu xanh. Vậy mô hình Linear Regression hoạt động tốt với tập dữ liệu training. Bây giờ, chúng ta sử dụng mô hình này để dự đoán cân nặng của hai người có chiều cao 155 và 160 cm mà chúng ta đã không dùng khi tính toán nghiệm.  
  ```
  y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )
  ```
```
Predict weight of person with height 155 cm: 52.94 (kg), real number: 52 (kg)
Predict weight of person with height 160 cm: 55.74 (kg), real number: 56 (kg)
```

  Ta nhận thấy rằng kết quả trả về gần giống với kết quả đã cho ở đề bài 


### Nghiệm theo thư viện scikit-learn

Tiếp theo, chúng ta sẽ sử dụng thư viện scikit-learn của Python để tìm nghiệm.
  
```
from sklearn import datasets, linear_model

# fit the model by Linear Regression
regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
regr.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', regr.coef_ )
print( 'Solution found by (5): ', w.T)
```
Kết quả :
```
    Solution found by scikit-learn  :  [[  -33.73541021 0.55920496]]
    Solution found by (5):  [[  -33.73541021 0.55920496 ]]
```
  
Chúng ta thấy rằng hai kết quả thu được như nhau!


### Hạn chế của Linear Regression  
Hạn chế đầu tiên của Linear Regression là nó rất nhạy cảm với nhiễu (sensitive to noise). Trong ví dụ về mối quan hệ giữa chiều cao và cân nặng bên trên, nếu có chỉ một cặp dữ liệu nhiễu (150 cm, 90kg) thì kết quả sẽ sai khác đi rất nhiều. Xem hình dưới đây:  
![anh](/Image/ptllinearnhieu.png)
  
 Vì vậy, trước khi thực hiện Linear Regression, các nhiễu (outlier) cần phải được loại bỏ. Bước này được gọi là tiền xử lý (pre-processing)  

Hạn chế thứ hai của Linear Regression là nó không biễu diễn được các mô hình phức tạp. Mặc dù trong phần trên, chúng ta thấy rằng phương pháp này có thể được áp dụng nếu quan hệ giữa outcome và input không nhất thiết phải là tuyến tính, nhưng mối quan hệ này vẫn đơn giản nhiều so với các mô hình thực tế. 

### Các phương pháp tối ưu
Linear Regression là một mô hình đơn giản, lời giải cho phương trình đạo hàm bằng 0 cũng khá đơn giản. Trong hầu hết các trường hợp, chúng ta không thể giải được phương trình đạo hàm bằng 0.

Nhưng có một điều chúng ta nên nhớ, còn tính được đạo hàm là còn có hy vọng.

[Code Linear Regression theo công thức ](/Code_LinearRegression/LinearRegression_func.py)
[Code Linear Regression theo scikit-learn](/Code_LinearRegression/LinearRegression_skl.py)
