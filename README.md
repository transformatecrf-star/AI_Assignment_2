Task 4: Rule-Based Baseline Summary

* Training Accuracy:25.89% (compared to the 29.07% constant baseline floor)
* Core Limitation: Deterministic keyword rules suffered from high lexical ambiguity, generating a high volume of out-of-class false positives that inadvertently overrode correct majority-class (*Technical Support*) predictions.
* Engineering Takeaway: Rigid heuristics cannot capture semantic context in noisy text datasets. This negative result empirically justifies transitioning to statistical, probabilistic models (Task 5) like TF-IDF and Logistic Regression.
