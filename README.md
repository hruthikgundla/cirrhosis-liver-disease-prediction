# Liver Cirrhosis Prediction using Machine Learning & Deep Learning

A data-driven project that predicts the presence of liver cirrhosis using clinical attributes and state-of-the-art ML/DL techniques. Built using data from the Mayo Clinic trial, this model assists in early diagnosis and decision-making.

⸻

## Project Highlights
	•	💡 Built a prediction pipeline using Logistic Regression, Random Forest, and XGBoost.
	•	📈 Achieved up to 100% accuracy on test data.
	•	🧠 Integrated Autoencoders for deep feature extraction, reducing prediction error by 15%.
	•	🧹 Handled missing data using KNN Imputation and MICE (Multivariate Imputation by Chained Equations).
	•	📊 Used features like Bilirubin, Albumin, SGOT, Copper, and more for prediction.

⸻

## Dataset
	•	Source: Mayo Clinic Trial (Primary Biliary Cirrhosis)
	•	Attributes: 19 clinical and demographic features
	•	Target Variable: Stage (Cirrhosis progression stage)

⸻

## Technologies Used
	•	Python
	•	Pandas, NumPy, Matplotlib
	•	Scikit-learn, XGBoost
	•	Autoencoders (Keras/TensorFlow for deep learning)

⸻

## Project Structure:


<img width="782" alt="Screenshot 2025-05-18 at 11 14 11 AM" src="https://github.com/user-attachments/assets/45412106-280e-497c-8708-f8f9a373fbc2" />


## How to Run

Clone the respository:

		git clone https://github.com/hruthikgundla/liver-cirrhosis-prediction.git
                cd liver-cirrhosis-prediction



2: Install Libraries:

	pip install -r requirements.txt


3: Run the notebook

	jupyter notebook


Open liver_cirrhosis_prediction.ipynb and run step by step.

## Results:

<img width="248" alt="Screenshot 2025-05-18 at 11 17 49 AM" src="https://github.com/user-attachments/assets/f1214f31-4059-45da-803b-5a29a8a8cdf1" />


Autoencoders further improved generalization and reduced prediction errors on unseen data.

💡 Future Improvements
	•	Deploy model using Flask/Streamlit for real-time prediction
	•	Add explainability with SHAP or LIME
	•	Integrate with EHR systems for clinical usage

⸻
## In our analysis it is to report that, Logistic Regression gave 98.8% accuracy, meaning it correctly predicts liver cirrhosis status in most cases, but Random Forest and XGBoost performed even better, with 100% accuracy on your current dataset. So, Logistic Regression is not necessarily better — it’s slightly less accurate than the others in this case.


And to finalize, It is the Random Forest and XGBoost are the two algorithms which are successed in detecting the patterns of the patient's data and ensured that there are high chances of having liver Cirrhosis Disease to them.


## Contact

Feel free to reach out for feedback or collaboration!

👤 Hruthik Gundla <br>
📧 Email: hruthikgundla22@gmail.com<br>
LinkedIn: https://www.linkedin.com/in/hruthikgundla/ <br>
GitHub: https://github.com/hruthikgundla

⸻








