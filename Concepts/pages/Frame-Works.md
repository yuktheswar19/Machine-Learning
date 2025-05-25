# **Different Approaches of Machine Learning**
	- Supervised  Learning
	- Un-supervised Learning
	- Reinforcement learning
	-
	- ## **Supervised Learning**
		- Supervised learning is an approach in machine learning where algorithms are trained based on the labelled datasets . In dataset each input is paired with the correct output , allowing the model to learn the relationsip between inputs and outputs
			- Inputs must be called as - **Features**
			- Outputs are called as the - **Targets**
		- **Training Process :**
		  collapsed:: true
			- The model processes the data , by learning which means mapping the  features to the corresponding targets . By minimising errors between predictions and the actual data labels
			- After the training the model is tested with the new unseen data for predictions
		- Examples :
		  collapsed:: true
			- Email spam detection
			- Image Recognition
			- Fraud Detection
				- We us the data sets which contain the Features as well as the target
			-
		- # **Algorithms**
			- ## **Regression Tasks**
				- ### **Linear Regression**
				  collapsed:: true
					- Linear Regression is a statistical technique use to determine the relation ship between two different variables , which they are called as Independent variable and the dependent variable
						- **Dependent variable** - (The outcome or prediction)
						- **Independent variable** - (Input features for prediction)
					- In simpler terms it is an analysis which is used to predict the value of a variable based of off another variable .
						- The variable you want to predict is called dependent variable
						- The variable you are using to predict the other variable's value is called the independent variable.
					- It is to find the relation ship between 2 variables based on the features
					- ```
					  y = mx + c
					  ```
					- where y is the predicted value, x is the independent variable, mm is the slope, and cc is the y-intercept###
					- ### **What is Linear Regression Used for?**
						- **Predict outcomes:** For example, predicting sales based on advertising spend, or predicting house prices based on size
						- **Understand relationships:** Quantify how much one 
						  variable changes in response to another, such as how temperature affects crop yields or how price affects product sales
						- **Optimise decisions:** Help businesses and researchers make informed decisions, such as setting optimal prices or identifying risk factors in medicine
						-
				- ### **Multiple Linear Regression**
				  collapsed:: true
					- This is also similar to linear regression the code difference between linear and multiple is that , multiple linear regression focuses on finding the dependent variable with multiple independent variables
					- ```
					  y=  β0​ + β1​X1​ + β2​X2​ + ⋯ + βn​Xn
					  ```
					- ### **Scenario Example for better understanding Linear and Multiple linear**
						- ### **Simple Linear Regression Example**
							- Suppose you want to predict the price of a house based only on its size (in square feet)
								- **Independent variable (X):** Size of the house (sq ft)
								- **Dependent variable (Y):** Price of the house
							- **Interpretation:**
								- If you know the size of a house, you can use this equation to estimate its price. For example, if the model finds that every additional square foot increases the price by $200, and the base price is $50,000, then a 1,000 sq ft house would be predicted
								- ```
								  Price=50,000+200×1,000=250,000
								  ```
						- ### **Multiple Linear Regression Example**
							- **Independent variables (X1, X2, X3)**
								- Size of the house (sq ft)
								- Number of bedrooms
								- Age of the house (years)
							- **Dependent variable (Y):** Price of the house
							- ```
							  Price= b0 + b1×Size + b2×Bedrooms + b3×Age
							  ```
							- **Interpretation:**
								- Now, the model will estimate how much each factor (size, bedrooms, age) contributes to the price, holding the others constant. For example:
									- Every additional square foot might increase the price by $180.
									- Each extra bedroom might add $10,000.
									- Each year of age might decrease the price by $1,500.
								- So, for a 1,000 sq ft, 3-bedroom, 10-year-old house:
									- ```
									  Price=40,000 +180×1,000 + 10,000× 3−1,500×10 = 40,000+180,000+30,000−15,000=235,000
									  ```
				- ### **Polynomial Regression**
				  collapsed:: true
					- Polynomial regression is similar to the linear regression but during prediction the curve is not a straight line but a curved line . Polynomial regression fits the curved line by including higher power of the independent variable
						- **Simple liner regression :**
							- ```
							  y=β0+β1x+ε
							  ```
						- **polynomial regression**:
							- ```
							  y=β0+β1x+β2x2+β3x3+…+βnxn+ε
							  ```
					- ### **Simple examples to understand :**
						- Suppose you're trying to predict the price of a car based on its age. If the price drops quickly in the first few years and then levels off, the relationship between age and price is not a straight line. A simple linear regression would not fit this pattern well. Polynomial regression, by adding x2x2 or x3x3 terms, can model this curved relationship more accurately
						- modeling the growth of a plant over time, where growth accelerates and 
						  then slows down, or predicting sales that rise and fall over seasons.
				- ### **Support Vector Regression**
				  collapsed:: true
					- Support vector regression is a machine learning algorithm used to predict continuous values such as temperature or prices rather than categories. It works by creating a "tube" (called the **epsilon-insensitive tube**) around the predicted regression line. Points inside this tube are ignored, and only points outside the tube (called **support vectors**) influence the model’s accuracy
						- https://cdn-images-1.medium.com/max/892/1*M57OgznesBrXu2WcpfpTGA.jpeg
					-
						- **Epsilon (ε) Tube**: A margin around the regression line where errors are tolerated. Predictions within this tube are considered acceptable.
						- **Support Vectors**: Data points outside the tube that define the tube’s position and width.
						- **Slack Variables (ξ)**: Penalties for points outside the tube. The model tries to minimize these penalties.
					- **Simple Example to Understand:**
						- Suppose you use house size and age to predict price. SVR creates a tube around the predicted price line. If a house’s actual price is far outside this tube (e.g., due to unique features), it becomes a support vector, and the model adjusts to account for it.
						- When you apply SVR, the algorithm doesn’t try to fit every single data point exactly; instead, it creates a margin of tolerance (called an epsilon tube) around the predicted price line. Most houses with typical prices for their size and features will fall within this tube, and small errors inside this margin are ignored. This means the model is not overly sensitive to minor variations in house prices that are considered normal.
						- houses with prices that are unusually high or low compared to similar houses—perhaps due to luxury upgrades or poor condition—fall outside the tube and become support vectors. These outlier houses influence the shape and position of the prediction line, helping the model adjust so that it is not overly affected by unusual cases but still learns from them.
				- ### **Decision Tree Regression**
				  collapsed:: true
					- Decision tress is a  supervised machine learning algorithm is used to classify and regress the data using true or false answers to certain questions
					- **Decision tree regression** is a supervised machine 
					  learning method used to predict continuous numerical values. It works by splitting the dataset into smaller and smaller subsets based on certain features, forming a tree-like structure composed of decision nodes (where a split happens) and leaf nodes (which represent the final 
					  predicted value). At each decision node, the algorithm chooses a feature and a threshold that best separates the data into groups with similar target values, aiming to make each group (or branch) as homogeneous as possible
					- ### **Taking an Example to understand**
					  collapsed:: true
						- https://pplx-res.cloudinary.com/image/upload/v1748072638/gpt4o_images/lpyveosizlhsrjgthqr9.png
					- collapsed:: true
						- To predict the price of a house using a decision tree regression model. The features you have include the size of the house (in square feet), the number of bedrooms, the location, and the age of the house.
						- The decision tree starts at the root with all the training data. It looks for the feature and value that best splits the data to minimize the difference in house prices within each group. For example, the tree might first split on "Size > 2000 sq ft?" If yes, the data goes to the right branch; if no, it goes to the left
							- **Left branch** (houses ≤ 2000 sq ft): The tree might next split on "Number of bedrooms > 3?"
							- **Right branch** (houses > 2000 sq ft): The tree might split on "Location = City Center?"
						- This process continues, with each branch splitting further based on other features like the age of the house or whether it has a garden, until the tree reaches a point where further splitting does not significantly improve the prediction
				- ### **Random Forest Regression**
				  collapsed:: true
					- Random forest regression is an ensemble method that combines the predictions from multiple decision tress to produce a mode accurate and stable prediction . It is a supervised machine learning algorithm for both classification and regression
					- In regression task we can use **Random Forest Regression** technique for predicting numerical values. It predicts continuous values by averaging the results of multiple decision trees.
					- ![Detailed illustration of Random Forest Regression with house price prediction example](https://pplx-res.cloudinary.com/image/upload/v1748073092/gpt4o_images/ude5mvn0jzrrsfer9lmz.png)
					-
						- we are using random forest regression to predict the price of a house based on features like size, number of bedrooms, location, and age
						- **Building Multiple Trees:**
							- The algorithm creates several decision trees, each using a random subset of the data and a random subset of the features. For example, one tree 
							  might focus on size and location, while another might focus on bedrooms and age. Each tree learns different patterns and rules from the data
						- **Making Predictions:**
							- When you want to predict the price of a new house, you input its features (e.g., 1800 sq ft, 3 bedrooms, city center, 5 years old) into every tree in the forest. Each tree independently predicts a price basedon the splits and rules it learned during training. For instance, Tree 1might predict ₹1.45 crore, Tree 2 might predict ₹1.55 crore, and Tree 3 might predict ₹1.50 crore
						- **Averaging the Results:**
							- The final predicted price is the average of all the individual tree predictions. This averaging helps cancel out errors or biases from any single tree, leading to a more reliable and accurate estimate. So, if the three trees above predicted ₹1.45 crore, ₹1.55 crore, and ₹1.50 crore, the final prediction would be ₹1.50 crore.
- test/ss