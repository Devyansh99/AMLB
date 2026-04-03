The Difference in Simple English

   * model_v3_classification.ipynb (The First Try):
       * What it is: This was your "Proof of Concept." You realized that predicting an exact score (1, 2, 3, 4, 5)
         wasn't working, so you switched to a simple "Happy vs. Unhappy" (Binary) model.
       * What it does: It uses basic features (like price and state) and a standard XGBoost model to see if the idea of
         a "Red Flag" detector was even possible.

   * v3_updated.ipynb (The Pro Version):
       * What it is: This is the much smarter, "production-ready" version.
       * What it does:
           1. Smarter Memory: It uses Target Encoding. Instead of just giving a random number to a Seller ID, it teaches
              the model which sellers usually have unhappy customers.
           2. Preparation Logic: It adds a feature called prep_to_delivery_ratio. This tells the model if the delay
              happened because the Seller was slow to pack it or the Carrier was slow to drive it.
           3. The "Safety Net": It doesn't just use a 50/50 guess. It finds the "Optimal Threshold" to make sure you
              catch as many "Red Flags" as possible, even if the model is only 51% sure.



 What’s inside suggestion.ipynb:

   1. Haversine Distance (MMBazel Repo): It uses distance_km to account for the physical distance between the seller and
      the buyer.
   2. Logistics Gap Analysis (Camille-Le-Roy Repo): It calculates the delivery_delta (the difference between when the
      package was promised and when it actually arrived).
   3. Advanced Target Encoding (Your V3 Pro): It teaches the model to recognize which seller_id and
      product_category_name are high-risk without needing thousands of dummy columns.
   4. Risk-Adjusted Threshold (Your V3 Pro): It doesn't just guess "Yes/No." It allows you to tune the model to catch
      more "Red Flags," which is exactly what a real business like Olist would do to prevent bad reviews.




The results for Class 1 (Dissatisfied) are lower because of a very common problem in Machine Learning called "Class
  Imbalance."

  Here is the simple reason why:

  1. The "4-to-1" Problem
  In your dataset, about 79% of customers are Happy (Class 0) and only 21% are Unhappy (Class 1).
   * Imagine a teacher has 100 students. 80 are always on time, and 20 are late.
   * If the teacher just guesses "Everyone will be on time," they will be 80% correct without even trying!
   * The model "prefers" to guess Class 0 because it's the safest bet to get a high accuracy score.

  2. Logistics vs. Reality (Missing Info)
  The model only knows about Shipping (how long it took, how far it went). But people leave 1-star reviews for many
  reasons that the model cannot see:
   * The product was broken.
   * The color was wrong.
   * The quality was bad.
   * The seller was rude in messages.
  Because the model is "blind" to these quality issues, it struggles to catch 100% of the unhappy people.

  3. Human Nature (The "Noise")
   * Some people get their package 5 days late and still leave a 5-star review because they are nice.
   * Some people get their package 2 days early but leave a 1-star review because they are having a bad day.
  This "human randomness" makes it very hard for a computer to be 100% accurate.

  How we fixed this in the notebooks:
  In v3_updated.ipynb and suggestion.ipynb, we used two "tricks" to help:
   1. scale_pos_weight: We told the XGBoost model: "Hey, every time you miss a Class 1 customer, it counts as 4 mistakes
      instead of 1." This forces it to pay more attention to them.
   2. Threshold Tuning: We moved the "guessing line" so the model is more willing to say "This looks like a Red Flag"
      even if it's only 40-50% sure.

  In short: Class 1 is harder because there are fewer examples of it, and because people's feelings are more complicated
  than just shipping speeds.


- get insight which is really helpful for the data
- get more conceptual point, the data is not rich but we can do findings.
- some features which are not in sellers control and re-rank the list