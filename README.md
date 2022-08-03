# Task

The UCI ML Drug Review dataset provides patient reviews on specific drugs along with related conditions and a 10-star patient rating system reflecting overall patient satisfaction. The data was obtained by crawling online pharmaceutical review sites. This data was published in a study on sentiment analysis of drug experience over multiple facets, ex. sentiments learned on specific aspects such as effectiveness and side effects.
There is no specific task that needs to be solved. You may do whatever you want with data. Here are just a couple ideas:
Classification: Can you predict the patient's condition based on the review?
Regression: Can you predict the rating of the drug based on the review?
Sentiment analysis: What elements of a review make it more helpful to others? Which patients tend to have more negative reviews? Can you determine if a review is positive, neutral, or negative?
Data visualizations: What kind of drugs are there? What sorts of conditions do these patients have?
Feel free to think any other task. The goal is to show your skills. The only condition is to leave comments describing your steps and clear code.

To run:
- Install dependencies
- Run from terminal using the command
```
pip install requirements.txt
python3 ml.py
```

## Summary of results and future improvement
LGBM classifier reports the best accuracy of 83%. XGBoost and MultinomialNB both have an accuracy of 77%.
MultinomialNB has a 93% recall call which shows a large number of false negatives.

The models have a good performance but can be further improved by
1. More feature engineering: We can extract features like parts of speech, abstraction [1], paragraphs,
    number of sentences in each text, sentimental biases in patients review
2. Hyperparameter tuning using methods like grid search, cross validation....
3. Use different discrete classifier algorithms

References

[1] Johnson-Grey KM, Boghrati R, Wakslak CJ, Dehghani M. Measuring Abstract Mind-Sets Through Syntax:
    Automating the Linguistic Category Model. Social Psychological and Personality Science. 2020;11(2):217-225.
    doi:10.1177/1948550619848004