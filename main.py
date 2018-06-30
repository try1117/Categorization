from shared import *
from classifier import *
from feature_extractor import *
from classification_algorithm import *
import gc

def main():
    cat_cnt = 30
    ctg = Classifier()
    ctg.read("data/utf-8/2_final_tables/products_dns_short_sorted_utf8.csv", "data/utf-8/2_final_tables/categories_utf8.csv", cat_cnt=cat_cnt)
    # ctg.read_opponents("data/utf-8/2_final_tables/mega_sub_1e6_filled_07.csv")
    # ctg.read_opponents("data/utf-8/2_final_tables/mega_sub_filled_02.csv")
    # quit()

    # word2vec_extractor = Word2VecFeatureExtractor().initialize(ctg.descriptions, ctg.total_descriptions_cnt, 300, 4, 3, 2, 1e-3)
    word2vec_extractor = Word2VecFeatureExtractor().load("word2vec_models/features=300_mincount=3_context=2")
    # print(len(word2vec_extractor.index2word_set))
    # print(word2vec_extractor.index2word_set)
    # quit()

    unigram_word = NGramFeatureExtractor("word", (1, 1), 1000, DescriptionPreprocessor.simple_processing)
    bigram_word = NGramFeatureExtractor("word", (2, 2), 1000, DescriptionPreprocessor.simple_processing)
    biuni_word = NGramFeatureExtractor("word", (1, 2), 1000, DescriptionPreprocessor.simple_processing)
    bigram_char = NGramFeatureExtractor("char", (2, 3), 1000, DescriptionPreprocessor.special_processing)
    bigram_char_spaces = NGramFeatureExtractor("char", (2, 3), 1000, DescriptionPreprocessor.simple_processing)

    ctg.draw_categories([10, 25], word2vec_extractor)
    quit()

    # cr_bay = ctg.cross_validate(4, 0.75, 1000, word2vec_extractor, NaiveBayesAlgorithm(), 2)
    # cr_bay.save("output/bay_cat=30_1000_word2vec_features=300.xlsx")
    # quit()

    # cr_neigh = ctg.cross_validate(1, 0.75, 1000, bigram_char, KNeighborsAlgorithm(3), 2)
    # cr_neigh.save("output/neigh=3_cat=30_char_features=1000.xlsx")
    # quit()

    # cr_regr = ctg.cross_validate(4, 0.75, 1000, unigram_word, LogRegressionAlgorithm(0.001))
    # cr_regr.save("output/regr_tol=0.001_cat=100_word_features=1000.xlsx")
    # quit()

    # RANDOM FOREST

    # cr_forest, predictor = ctg.cross_validate(1, 0.75, 1000, biuni_word, RandomForestAlgorithm(10), 2)
    # cr_forest.save("output/prez/forest_cat={}_bigram_char.xlsx".format(cat_cnt))

    # cr_forest, predictor = ctg.cross_validate(1, 0.75, 1000, bigram_char, RandomForestAlgorithm(10), 2)
    # cr_forest.save("output/prez/forest_cat={}_bigram_char.xlsx".format(cat_cnt))

    # cr_forest, predictor = ctg.cross_validate(1, 0.75, 1000, bigram_char, RandomForestAlgorithm(10), 1)
    # ctg.test_opponents(bigram_char, predictor)
    # quit()
    # cr_forest.save("output/forest_cat=40_word2vec_features=300.xlsx")
    # quit()

    # cr_forest = ctg.cross_validate(4, unigram_word, RandomForestAlgorithm(10))
    # cr_forest.save("output/forest.xlsx")
    # quit()

    # algorithms = {"forest": RandomForestAlgorithm(10), "logregr": LogRegressionAlgorithm()}
    features = {"biuni_word": biuni_word, "bigram_char": bigram_char, "word2vec": word2vec_extractor}
    # features = {"word2vec": word2vec_extractor}
    # algorithms = {"bayess": NaiveBayesAlgorithm()}
    algorithms = {"neighbor": KNeighborsAlgorithm(1)}

    for f_key, f_value in features.items():
        for a_key, a_value in algorithms.items():
            print("\n\nSTART {} {}\n\n".format(f_key, a_key))
            cv_res, predictor = ctg.cross_validate(1, 0.75, 1000, f_value, a_value)
            cv_res.save("output/prez/cat={}_{}_{}.xlsx".format(cat_cnt, f_key, a_key))
            print("\n\nDONE {} {}\n\n".format(f_key, a_key))
            gc.collect()

if __name__ == "__main__":
    main()
