**Predict a filling strategy for a route**
----
  Fetch the optimal filling strategy based on a path and the filling stations near it that are known to the app.
  
* **URL**

  ```prediction/google```

* **Method:**
  
  `POST`
  
*  **URL Params**

   none

* **Data Params**

  `json`-File
  
  **Must contain**
  * `path` : `[[long_start,lat_end],[long,lat],...]` # the path f the route approximated as a line
  * `length` : 42 # the length of the path in kilometers
  * `speed` : 0 # the speed of the car in kmh
  * `capacity`: 50 # the capacity of the tank in liters
  * `fuel` : 10 # the fuel a car has in the beginning of te journey
  

* **Success Response:**
  
  Algorithm terminated and found a result.
  
  * **Code:** 200 <br />
    **Content:** `{'start': [4,54] - startcoordinate,
            'end': [6,52] - endcoordinate,
            'stops': [[6,51],[7,53],...] - coordinates on filling stations on way,
            'prices': [1.12,1.23,1.10,...] - predicted prices at those stations,
            'fill_liters': [3,4.23,.0,...] - liters that musst be filled up,
            'payment': [12.4,12.5,12.4,...] - price tat will be paid at gas_station.
            'overall_price': 100.23 - overall price}`
 
* **Error Response:**

  * **Code:** 418 I'm a teapot <br />
    **Content:** `{ error : "No JSON has been provided" }`

  OR
  
  * **Code:** 418 I'm a teapot <br />
    **Content:** `{ error : "Also expected this key(s): {}" }`

  * **Code:** 500 Internal Server Error <br />
    **Content:** `{ error : "Algorithm failed" }`
