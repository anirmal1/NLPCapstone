import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from fnc_1_baseline_master.feature_engineering import reg_counts, discuss_features, refuting_features, polarity_features, hand_features, gen_or_load_feats
from fnc_1_baseline_master.feature_engineering import word_overlap_features, LIWC_lexicons, gen_or_load_feats_liwc
from fnc_1_baseline_master.utils.dataset_svm import DataSet
from fnc_1_baseline_master.utils.generate_test_splits import kfold_split, get_stances_for_folds
from fnc_1_baseline_master.utils.score import report_score, LABELS, score_submission
from fnc_1_baseline_master.utils.system import parse_params, check_version
# from fnc_1_baseline_master.LIWC.LIWCutil import extract, reverse_dict

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    # Get LIWC lexicons
    liwc_lex = LIWC_lexicons('2015')

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "fnc_1_baseline_master/features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "fnc_1_baseline_master/features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "fnc_1_baseline_master/features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "fnc_1_baseline_master/features/hand."+name+".npy")
    X_discuss = gen_or_load_feats(discuss_features, h, b, "fnc_1_baseline_master/features/discuss."+name+".npy")
    # X_pronoun = gen_or_load_feats_liwc(reg_counts, liwc_lex['pronoun'], h, b, "fnc_1_baseline_master/features/pronoun_reg."+name+".npy")
    # X_anx = gen_or_load_feats_liwc(reg_counts, liwc_lex['anx'], h, b, "fnc_1_baseline_master/features/anx_reg."+name+".npy")
    # X_anger = gen_or_load_feats_liwc(reg_counts, liwc_lex['anger'], h, b,  "fnc_1_baseline_master/features/anger_reg."+name+".npy") 
    # X_negate = gen_or_load_feats_liwc(reg_counts, liwc_lex['negate'], h, b, 'fnc_1_baseline_master/features/negate_reg.'+name+'.npy')
    # X_quant = gen_or_load_feats_liwc(reg_counts, liwc_lex['quant'], h, b, 'fnc_1_baseline_master/features/quant_reg.'+name+'.npy')

    X = np.c_[X_discuss, X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

if __name__ == "__main__":
    check_version()
    parse_params()

    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    for fold in fold_stances:
        Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))


    best_first_score = 0
    best_first_fold = None
    best_second_score = 0
    best_second_fold = None

    # Classifier for each fold
    for fold in fold_stances:
        print('Fold ' + str(fold) + '...')

        ids = list(range(len(folds)))
        del ids[fold]


        # SVM 1 : distinguishing unrelated from related
        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        y_train_round1 = []
        for item in y_train:
          if item == LABELS.index('unrelated'):
            y_train_round1.append(0)
          else:
            y_train_round1.append(1)
        y_train_round1 = np.array(y_train_round1)

        X_test = Xs[fold]
        y_test = ys[fold]

        y_test_round1 = []
        for item in y_test:
          if item == LABELS.index('unrelated'):
            y_test_round1.append(0)
          else:
            y_test_round1.append(1)
        y_test_round1 = np.array(y_test_round1)

        clf1 = svm.SVC()
        clf1.fit(X_train, y_train_round1)

        round1_pred = clf1.predict(X_test)
        round1_score = 0
        for i in range(len(round1_pred)):
          if round1_pred[i] == y_test_round1[i]:
            round1_score += 1
        round1_score = 1.0 * round1_score / len(round1_pred)

        print('round 1 score: ' + str(round1_score))

        # SVM 2 - distinguishing categories within related
        X_train_round2 = []
        y_train_round2 = []
        for i in range(len(X_train)):
          if y_train[i] != LABELS.index('unrelated'):
            X_train_round2.append(X_train[i])
            y_train_round2.append(y_train[i])

        X_test_round2 = []
        y_test_round2 = []
        for i in range(len(X_test)):
          if y_test[i] != LABELS.index('unrelated'):
            X_test_round2.append(X_test[i])
            y_test_round2.append(y_test[i])

        clf2 = svm.SVC()
        clf2.fit(X_train_round2, y_train_round2)

        round2_pred = clf2.predict(X_test_round2)
        round2_score = 0
        for i in range(len(round2_pred)):
          if round2_pred[i] == y_test_round2[i]:
            round2_score += 1
        round2_score = 1.0 * round2_score / len(round2_pred)

        print('round 2 score: ' + str(round2_score))
        
        # fold_score, _ = score_submission(actual, predicted)
        # max_fold_score, _ = score_submission(actual, actual)

        # score = fold_score/max_fold_score

        # print("Score for fold "+ str(fold) + " was - " + str(score))
        if round1_score > best_first_score:
            best_first_score = round1_score
            best_first_fold = clf1
        if round2_score > best_second_score:
            best_second_score = round2_score
            best_second_fold = clf2


    #Run on Holdout set and report the final score on the holdout set
    # predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    predicted = []
    predicted_1 = [a for a in clf1.predict(X_holdout)]
    for i in range(len(predicted_1)):
      item = predicted_1[i]
      if item == 0: # 0 is the code for unrelated
        predicted.append(LABELS.index('unrelated'))
      else:
        predicted_2 = clf2.predict([X_holdout[i]])
        predicted.append(predicted_2)

    actual = [LABELS[int(a)] for a in y_holdout]

    report_score(actual,predicted)
