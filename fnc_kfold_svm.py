import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import metrics
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
    X_negate = gen_or_load_feats_liwc(reg_counts, liwc_lex['negate'], h, b, 'fnc_1_baseline_master/features/negate_reg.'+name+'.npy')
    # X_quant = gen_or_load_feats_liwc(reg_counts, liwc_lex['quant'], h, b, 'fnc_1_baseline_master/features/quant_reg.'+name+'.npy')

    X = np.c_[X_negate, X_discuss, X_hand, X_polarity, X_refuting, X_overlap]
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


    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        #clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
        #clf.fit(X_train, y_train)

        clf = svm.SVC()
        clf.fit(X_train, y_train)

        predicted = [LABELS[int(a)] for a in clf.predict(X_test)]
        actual = [LABELS[int(a)] for a in y_test]
        
	# Mean F1 score per fold
        f1_score = metrics.f1_score(actual, predicted, average='macro')
        print("F1 MEAN score for fold " + str(fold) + " was - " + str(f1_score))

        # F1 score each label
        f1_score_labels = metrics.f1_score(actual, predicted, labels=LABELS, average=None)
        print("F1 LABEL score for fold " + str(fold) + " was - " + str(f1_score_labels))
        
	# FNC scoring metric
        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf


    #Run on Holdout set and report the final score on the holdout set
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]

    f1_score = metrics.f1_score(actual, predicted, average='macro')
    print("F1 MEAN score overall " + str(fold) + " was - " + str(f1_score))
    
    report_score(actual,predicted)
