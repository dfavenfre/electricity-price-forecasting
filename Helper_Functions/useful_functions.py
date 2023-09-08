def pre_process():
  """
  Description:
  ------------
  Input data is cleaned using the following methodologies:

  * Date and hour data are separated in the original dataset, and datetime formatting has been applied as %d%m/%Y+%H:+%M (day, month, year: hour, minute).
  * The concatenated datetime object is then indexed.
  * Unnecessary columns such as "Tarih" and "Saat" are dropped since the data now has the desired datetime format.
  * Certain variables are renamed, such as "Pozitif Dengesizlik Miktarý (MWh)" renamed as "positive_imbalance" or "PTF (TL/MWh)" as "PTF."
  * Regular expressions are used to clear the incorrect numeric formatting, and all numeric variables are then converted to float type.
  * The "imbalance_delta" variable is derived from the addition of "positive_imbalance" and "negative_imbalance."
  * Finally, all the data is converged into one entity.

  """

  # imbalance_delta 
  imbalance_delta['Tarih'] = pd.to_datetime(imbalance_delta['Tarih'], format='%d/%m/%Y')
  imbalance_delta['Saat'] = pd.to_datetime(imbalance_delta['Saat'], format='%H:%M').dt.time
  imbalance_delta["datetime"] = imbalance_delta['Tarih'].dt.strftime('%d/%m/%Y') + ':' + imbalance_delta['Saat'].astype(str)
  imbalance_delta.rename(columns={
      "Pozitif Dengesizlik Miktarý (MWh)": "positive_imbalance",
      "Negatif Dengesizlik Miktarý (MWh)":"negative_imbalance"},
      inplace=True)
  imbalance_delta.drop(["Tarih","Saat"],axis=1, inplace=True)
  imbalance_delta['positive_imbalance'] = imbalance_delta['positive_imbalance'].str.replace('.', '').str.replace(",",".").astype(float)
  imbalance_delta['negative_imbalance'] = imbalance_delta['negative_imbalance'].str.replace('.', '').str.replace(",",".").astype(float)
  imbalance_delta.set_index("datetime",inplace=True)

  # ptf
  ptf['Tarih'] = pd.to_datetime(ptf['Tarih'], format='%d/%m/%Y')
  ptf['Saat'] = pd.to_datetime(ptf['Saat'], format='%H:%M').dt.time
  ptf["datetime"] = ptf['Tarih'].dt.strftime('%d/%m/%Y') + ':' + ptf['Saat'].astype(str)
  ptf.rename(columns={
      "PTF (TL/MWh)": "PTF"},
      inplace=True)
  ptf.drop(["Tarih","Saat","PTF (USD/MWh)", "PTF (EUR/MWh)"],axis=1, inplace=True)
  ptf['PTF'] = ptf['PTF'].str.replace('.', '').str.replace(",",".").astype(float)
  ptf.set_index("datetime",inplace=True)  

  # ask_amount
  ask_amount['Tarih'] = pd.to_datetime(ask_amount['Tarih'], format='%d/%m/%Y')
  ask_amount['Saat'] = pd.to_datetime(ask_amount['Saat'], format='%H:%M').dt.time
  ask_amount["datetime"] = ask_amount['Tarih'].dt.strftime('%d/%m/%Y') + ':' + ask_amount['Saat'].astype(str)
  ask_amount.rename(columns={"Teklif Edilen Satýþ Miktarý (MWh)":"ask_amount"},inplace=True)
  ask_amount.drop(["Tarih","Saat"],axis=1, inplace=True)  
  ask_amount['ask_amount'] = ask_amount['ask_amount'].str.replace('.', '').str.replace(",",".").astype(float)
  ask_amount.set_index("datetime",inplace=True)

  # bid_amount
  bid_amount['Tarih'] = pd.to_datetime(bid_amount['Tarih'], format='%d/%m/%Y')
  bid_amount['Saat'] = pd.to_datetime(bid_amount['Saat'], format='%H:%M').dt.time
  bid_amount["datetime"] = bid_amount['Tarih'].dt.strftime('%d/%m/%Y') + ':' + bid_amount['Saat'].astype(str)
  bid_amount.rename(columns={"Teklif Edilen Alýþ Miktarý (MWh)":"bid_amount"},inplace=True)
  bid_amount.drop(["Tarih","Saat"],axis=1, inplace=True)  
  bid_amount['bid_amount'] = bid_amount['bid_amount'].str.replace('.', '').str.replace(",",".").astype(float)
  bid_amount.set_index("datetime",inplace=True)

  # volume
  volume['Tarih'] = pd.to_datetime(volume['Tarih'], format='%d/%m/%Y')
  volume['Saat'] = pd.to_datetime(volume['Saat'], format='%H:%M').dt.time
  volume["datetime"] = volume['Tarih'].dt.strftime('%d/%m/%Y') + ':' + volume['Saat'].astype(str)
  volume.rename(columns={"Ýþlem Hacmi (TL)":"volume"},inplace=True)
  volume.drop(["Tarih","Saat"],axis=1, inplace=True)
  volume['volume'] = volume['volume'].str.replace('.', '').str.replace(",",".").astype(float)
  volume.set_index("datetime",inplace=True)

  comb = pd.concat([imbalance_delta,ask_amount,bid_amount,volume,ptf],axis=1)
  comb = comb.reset_index()
  comb.drop(["datetime"],axis=1, inplace=True)
  comb["imbalance_delta"] = comb["positive_imbalance"] + comb["negative_imbalance"]
  comb.drop(["positive_imbalance","negative_imbalance"],axis=1, inplace=True)
  comb = comb.dropna()
  return comb


def evaluate_preds(y_test, y_pred):
  """
  Description:
  ------------
  Evaluates the performance of predicted values against the true values using various metrics.

  Parameters:
  -----------
  y_test: array-like
      True values of the target variable.
  y_pred: array-like
      Predicted values of the target variable.

  Returns:
  --------
  dict:
      A dictionary containing the calculated metrics, including Mean Absolute Error (MAE),
      Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
  """
  y_test = tf.cast(y_test, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  mae=tf.keras.metrics.mean_absolute_error(y_test, y_pred)
  mse=tf.keras.metrics.mean_squared_error(y_test, y_pred)
  rmse=tf.sqrt(mse)
  mape=tf.keras.metrics.mean_absolute_percentage_error(y_test, y_pred)

  return {"mae":mae.numpy(),
          "mse":mse.numpy(),
          "rmse":rmse.numpy(),
          "mape":mape.numpy()}
