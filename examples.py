# Example of using rec metrics
import easyinfo
import rec_metrics

num_verses = 4
num_queries = 2
len_true = 3
# num_verses = 31102
# num_queries = 7000
# len_true = 180
true_indices = np.random.randint(0, num_verses-1, (num_queries, len_true))
true_weights = np.random.uniform(size= (num_queries, len_true))
pred_relevances = np.random.uniform(size= (num_queries, num_verses))
true_indices = np.array([[1, 4, 2], [0, 1]])
true_weights = np.array([[.5, .25, .25], [.33, .2]])
pred_relevances = np.array([[.51, .251, .11, .511, .521], [.331, .21, .11, .11, .31]])
start()
rec_method = RecMetrics(k=[2, 3], metrics=['k', 'NDCG', 'Precision', 'Recall', 'F', 'MRR', 'Coverage', '#Coverage', 'PCoverage', '#PCoverage'], verbose=True)
rec_method(true_indices, true_weights, pred_relevances)
# rec_method.save('test_recmetric.pkl')
# rec_method.save_norms('recmetric_norms.csv')
# rec_method.load('test_recmetric.pkl')
# norms = rec_method.load_norms('recmetric_norms.csv')
# vprint(norms, 'loaded norms')
# from google.colab import files
# files.download('recmetric_norms.csv')
end("Calculated recommendation metrics")
