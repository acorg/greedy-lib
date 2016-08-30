def newpredictors():
    file = open("predictors.csv", "w")
    file.close
    file = open("predictors.csv", "r+")
    predictors = []
    options = []
    lastpredict = open("predictors_51.csv", "r")
    for line in lastpredict:
    	file.write(line)
    	predictors.append(line)
    for item in predictors:
        if len(item) > 5:
            items = item.replace(":", "_")
            items2 = items.replace("\n", "")
            new_options = items2.split(':' and '_')
            for new_item in new_options:
                if new_item not in predictors:
                    file.write("\n")
                    file.write(new_item)
                    predictors.append(new_item) 
    file.close()
    almost_predictors = []
    final_predictors = []
    for item in predictors:
    	output = item.replace("\n", "")
    	almost_predictors.append(output)
    possible_predictors = ['AT131', 'AT138', 'AV272', 'DE135', 'DE158', 'DE188', 'DE190', 'DG053', 'DG078', 'DG124', 'DG172', 'DG275', 'DN031', 'DN053', 'DN133', 'DN216', 'EG135', 'EK135', 'EK156', 'FS219', 'FY219', 'GK135', 'HN075', 'IM067', 'IN145', 'IR208', 'IS145', 'IT121', 'IT214', 'IV088', 'IV112', 'IV196', 'IV213', 'KN122', 'KN145', 'KR050', 'KR201', 'KR220', 'LQ226', 'LS157', 'NS045', 'NS133', 'NS145', 'NS193', 'NS209', 'NS278', 'NT262', 'NT276', 'PS227', 'PS289', 'PT143', 'SY219']
    for item in almost_predictors:
    	if len(item) == 5:
    		if item in possible_predictors:
    			final_predictors.append(item)
    	else:
    		final_predictors.append(item)
    print final_predictors

newpredictors()		