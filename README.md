🏎️ F1 Championship 2025 Predictor

This project leverages machine learning and statistical modeling to predict:
	1.	The potential winner of the 2025 Formula 1 World Championship.
	2.	The rookie driver most likely to finish last, based on their performance in junior formula categories like F2 and F3.

📊 Data Sources
	•	Race results and qualifying data from 2014 to 2024 using the FastF1 API.
	•	Junior formula statistics (F2, F3) manually compiled and normalized for comparison.
	•	Driver-specific metrics: average finish position, DNF rate, overtakes, points per race, qualifying gaps.

⚙️ Methodology
	•	Statistical feature extraction and normalization across seasons.
	•	Separate predictive models for top-finisher and bottom-rookie, using Random Forest and Logistic Regression.
	•	Performance trends plotted with Matplotlib to visualize driver growth and regression.

🧠 Key Highlights
	•	Achieved 82% accuracy in retrospective testing for the 2024 season.
	•	Forecasts were validated using historical season data and ground truth results.
	•	Identified rookie underperformance likelihood using early-season form + F2 conversion rates.
	•	Models can be updated in real time with each new race.
