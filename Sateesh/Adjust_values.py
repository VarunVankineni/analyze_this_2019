def adjust_value(confidence, default_ind, threshold):
    if default_ind == 0:
        return default_ind
    else:
        if confidence > threshold:
            return default_ind
        else:
            return (1-default_ind)
			
			

def probability_dataframe(model,test_data, predictions, threshold):
    probabilities = model.predict_proba(test_data)
    d = {'application_key':test['application_key'], 'default_ind':predictions.tolist(),\
         'probability':probabilities.tolist()}

    prediction_df = pd.DataFrame(data = d)
    #print(prediction_df)
    prediction_df['Adjusted_Values'] = prediction_df.apply(lambda row: adjust_value(row[2][row[1]], row[1],threshold)\
                                                          , axis = 1)
    prediction_df['Confidence'] = prediction_df.apply(lambda row: row[2][row[3]], axis =1)
    prediction_df.sort_values(by = 'Confidence', axis = 0, ascending=False, inplace = True)
    prediction_df.reset_index(drop=True, inplace = True)
    
    return prediction_df
	

	
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = train_X.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances



sum(probability_df[:10000]['Adjusted_Values'])

submission = probability_df[['application_key', 'Adjusted_Values']]
submission.to_csv('../Linearized_reduced_gridXGB.csv', index = False, header=False)