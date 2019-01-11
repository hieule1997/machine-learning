# Giới thiệu Machine learning

## Phân loại dựa trên phương thức học   
    - Supervise learning (Học có giám sát)  
    - Unsupervised learning ( Học không giám sát)  
    - Semi-supervised lerning ( Học bán giám sát)  
    - Reinforcement learning ( Học củng cố)  
### Supervise learning (Học có giám sát)
  #### Khái niệm
  Supervised learning là thuật toán dự đoán đầu ra (outcome) của một dữ liệu mới (new input) dựa trên các cặp (input, outcome) đã biết từ trước. Cặp dữ liệu này còn được gọi là (data, label), tức (dữ liệu, nhãn). 
  => Supervised learning là nhóm phổ biến nhất trong các thuật toán Machine Learning.
  ### Các thuật toán   
 #### Classification (Phân loại)   
    - Một bài toán được gọi là classification nếu các label của input data được chia thành một số hữu hạn nhóm. Ví dụ: Gmail xác định xem một email có phải là spam hay không; các hãng tín dụng xác định xem một khách hàng có khả năng thanh toán nợ hay không. Ba ví dụ phía trên được chia vào loại này.  
  #### Regression (Hồi quy)
    - Nếu label không được chia thành các nhóm mà là một giá trị thực cụ thể. Ví dụ: một căn nhà rộng x m2, có y phòng ngủ và cách trung tâm thành phố z km sẽ có giá là bao nhiêu?

    - Gần đây Microsoft có một ứng dụng dự đoán giới tính và tuổi dựa trên khuôn mặt. Phần dự đoán giới tính có thể coi là thuật toán Classification, phần dự đoán tuổi có thể coi là thuật toán Regression. Chú ý rằng phần dự đoán tuổi cũng có thể coi là Classification nếu ta coi tuổi là một số nguyên dương không lớn hơn 150, chúng ta sẽ có 150 class (lớp) khác nhau.   
 ### Unsupervise learning (Học có giám sát) 
 #### Khái niệm
    - Trong thuật toán này, chúng ta không biết được outcome hay nhãn mà chỉ có dữ liệu đầu vào. Thuật toán unsupervised learning sẽ dựa vào cấu trúc của dữ liệu để thực hiện một công việc nào đó
 #### Clustering (phân nhóm)  
    - Một bài toán phân nhóm toàn bộ dữ liệu X thành các nhóm nhỏ dựa trên sự liên quan giữa các dữ liệu trong mỗi nhóm. Ví dụ: phân nhóm khách hàng dựa trên hành vi mua hàng. Điều này cũng giống như việc ta đưa cho một đứa trẻ rất nhiều mảnh ghép với các hình thù và màu sắc khác nhau, ví dụ tam giác, vuông, tròn với màu xanh và đỏ, sau đó yêu cầu trẻ phân chúng thành từng nhóm. Mặc dù không cho trẻ biết mảnh nào tương ứng với hình nào hoặc màu nào, nhiều khả năng chúng vẫn có thể phân loại các mảnh ghép theo màu hoặc hình dạng.
 #### Association  
    - Là bài toán khi chúng ta muốn khám phá ra một quy luật dựa trên nhiều dữ liệu cho trước. Ví dụ: những khách hàng nam mua quần áo thường có xu hướng mua thêm đồng hồ hoặc thắt lưng; những khán giả xem phim Spider Man thường có xu hướng xem thêm phim Bat Man, dựa vào đó tạo ra một hệ thống gợi ý khách hàng (Recommendation System), thúc đẩy nhu cầu mua sắm  
### Semi-Supervised Learning (Học bán giám sát)
 #### Khái niệm  
    - Các bài toán khi chúng ta có một lượng lớn dữ liệu X nhưng chỉ một phần trong chúng được gán nhãn được gọi là Semi-Supervised Learning. Những bài toán thuộc nhóm này nằm giữa hai nhóm được nêu bên trên.   
### Reinforcement Learning (Học Củng Cố)  
 #### Khái niệm 
    - Reinforcement learning là các bài toán giúp cho một hệ thống tự động xác định hành vi dựa trên hoàn cảnh để đạt được lợi ích cao nhất (maximizing the performance). Hiện tại, Reinforcement learning chủ yếu được áp dụng vào Lý Thuyết Trò Chơi (Game Theory), các thuật toán cần xác định nước đi tiếp theo để đạt được điểm số cao nhất.


## Phân nhóm dựa trên chức năng  
 ### Regression Algorithms
   - Linear Regression
   - Logistic Regression
   - Stepwise Regression
 ### Classification Algorithms
   - Linear Classifier
   - Support Vector Machine (SVM)
   - Kernel SVM
   - Sparse Representation-based classification (SRC)
 ### Instance-based Algorithms
   - k-Nearest Neighbor (kNN)
   - Learning Vector Quantization (LVQ)
### Regularization Algorithms
   - Ridge Regression
   - Least Absolute Shrinkage and Selection Operator (LASSO)
   - Least-Angle Regression (LARS)
### Bayesian Algorithms
   - Naive Bayes
   - Gaussian Naive Bayes
### Clustering Algorithms 
   - k-Means clustering
   - k-Medians
   - Expectation Maximization (EM)
### Artificial Neural Network Algorithms
   - Perceptron
   - Softmax Regression
   - Multi-layer Perceptron
   - Back-Propagation
### Dimensionality Reduction Algorithms
   - Principal Component Analysis (PCA)
   - Linear Discriminant Analysis (LDA)
### Ensemble Algorithms
   - Boosting
   - AdaBoost
   - Random Forest