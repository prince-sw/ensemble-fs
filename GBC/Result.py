def write_results(file, results, k):
    for result in results.keys():
        file.write(result+","+str(k) +
                   ","+get_score(results[result]))
        

def get_score(results):
    ans = "{},{},{},{},{}\n".format(results["test_accuracy"].mean(),
                                    results["test_fmeasure"].mean(),
                                    results["test_precision"].mean(),
                                    results["test_recall"].mean(),
                                    results["test_roc"].mean())
    # ans = "{}\n".format(results["test_accuracy"].mean())
    return ans
