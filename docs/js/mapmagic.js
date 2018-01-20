
var map = null;
var boxpolys = null;
var directions = null;
var routeBoxer = null;
var distance = null; // km
var service = null;
var gmarkers = [];
var boxes = null;
var infowindow = new google.maps.InfoWindow();

function initialize() {
  // Default the map view to the continental U.S.
  var mapOptions = {
    center: new google.maps.LatLng(51, 10),
    mapTypeId: google.maps.MapTypeId.ROADMAP,
    zoom: 6
  };

  map = new google.maps.Map(document.getElementById("map"), mapOptions);
  service = new google.maps.places.PlacesService(map);

  routeBoxer = new RouteBoxer();

  directionService = new google.maps.DirectionsService();
  directionsRenderer = new google.maps.DirectionsRenderer({
    map: map
  });

  // If there are any parameters at eh end of the URL, they will be in  location.search
  // looking something like  "?marker=3"

  // skip the first character, we are not interested in the "?"
  var query = location.search.substring(1);

  // split the rest at each "&" character to give a list of  "argname=value"  pairs
  var pairs = query.split("&");
  for (var i = 0; i < pairs.length; i++) {
    // break each pair at the first "=" to obtain the argname and value
    var pos = pairs[i].indexOf("=");
    var argname = pairs[i].substring(0, pos).toLowerCase();
    var value = pairs[i].substring(pos + 1).toLowerCase();

    // process each possible argname  -  use unescape() if theres any chance of spaces
    if (argname == "to") {
      document.getElementById('to').value = unescape(value);
    }
    if (argname == "from") {
      document.getElementById('from').value = unescape(value);
    }
    if (argname == "dist") {
      document.getElementById('distance').value = parseFloat(value);
    }
    if (argname == "type") {
      document.getElementById('type').value = unescape(value);
    }
    if (argname == "keyword") {
      document.getElementById('keyword').value = unescape(value);
    }
    if (argname == "name") {
      document.getElementById('name').value = unescape(value);
    }
    if (argname == "submit") {
      route();
    }
  }

}

function computeTotalDistance(result) {
  var total = 0;
  var myroute = result.routes[0];
  for (var i = 0; i < myroute.legs.length; i++) {
    total += myroute.legs[i].distance.value;
  }
  total = total / 1000;
  return total;
}

function route() {
  // Clear any previous route boxes from the map
  clearBoxes();

  distance = parseFloat(document.getElementById("distance").value);

  var request = {
    origin: document.getElementById("from").value,
    destination: document.getElementById("to").value,
    travelMode: google.maps.DirectionsTravelMode.DRIVING
  };

  // Make the directions request
  directionService.route(request, function(result, status) {
    if (status == google.maps.DirectionsStatus.OK) {
      document.getElementById('side_bar').innerHTML = '';
      // todo clear markers
        map.markers = [];
        qmarkers = [];
      //directionsRenderer.setDirections(result);
      // Box around the overview path of the first route
      var path = result.routes[0].overview_path;
      //boxes = routeBoxer.box(path, distance);
      // alert(boxes.length);
      //drawBoxes();
      //findPlaces(0);
      //directionService.route(request2, function(result2, status) {if (status == google.maps.DirectionsStatus.OK) {directionsRenderer.setDirections(result2);}});
      // Do the back-end request
      var length = "100"; //todo
      var speed = $("#speed").val();
      var fuel = $("#fuel").val();
      var capacity = $("#tank").val();
      var startTime = $("#time").val();
      var mappedPath = new Array(path.length);
      for (var i = 0; i < mappedPath.length; i++) {
        mappedPath[i] = [path[i].lat(), path[i].lng()]
      }

      var routeData = {
          length: length,
          start_time: startTime,
          speed: speed,
          fuel: fuel,
          capacity: capacity,
          path: mappedPath
      };

      $.ajax({
          type: "POST",
          url: "/prediction/google",
          contentType: "application/json",
          dataType: "json",
          success: function (data,success) {
            // get waypoints of gas stations
            var waypoints = new Array(data['stops'].length);
            for (var i = 0; i < waypoints.length; i++) {
              var loc = new google.maps.LatLng(data['stops'][i][0],data['stops'][i][1]);
               waypoints[i] = {location: loc,stopover:false};
               createGasStationMarker(loc,data['prices'][i],data['fill_liters'][i],data["name"][i],data["address"][i]); //todo replace adress and name
            }
            var request2 = {
            origin: document.getElementById("from").value,
            destination: document.getElementById("to").value,
            travelMode: google.maps.DirectionsTravelMode.DRIVING,
            waypoints: waypoints};

            directionService.route(request2, function(result2, status) {if (status == google.maps.DirectionsStatus.OK) {directionsRenderer.setDirections(result2);}});
          },
          error: function (url, error, status) {
            alert("Error" + error);
          },
          data: JSON.stringify(routeData)
      });

    } else {
      alert("Directions query failed: " + status);
    }
  });
}

// Draw the array of boxes as polylines on the map
function drawBoxes() {
  boxpolys = new Array(boxes.length);
  for (var i = 0; i < boxes.length; i++) {
    boxpolys[i] = new google.maps.Rectangle({
      bounds: boxes[i],
      fillOpacity: 0,
      strokeOpacity: 1.0,
      strokeColor: '#000000',
      strokeWeight: 1,
      map: map
    });
  }
}


function findPlaces(searchIndex) {
  var type = document.getElementById('type').value;
  var keyword = document.getElementById('keyword').value;
  var name = document.getElementById('name').value;
  var request = {
    bounds: boxes[searchIndex],
  };
  if (!!type && (type != "")) {
    if (type.indexOf(',') > 0)
      request.types = type.split(',');
    else
      request.types = [type];
  }
  document.getElementById('side_bar').innerHTML += "boxes are: " + boxes + "<br>"
  if (!!keyword && (keyword != "")) request.keyword = keyword;
  if (!!name && (name != "")) request.name = name;
  service.radarSearch(request, function(results, status) {
    if (status == google.maps.places.PlacesServiceStatus.OK) {
      document.getElementById('side_bar').innerHTML += "bounds[" + searchIndex + "] returns " + results.length + " results<br>"
      for (var i = 0, result; result = results[i]; i++) {
        var marker = createMarker(result);
      }
    } else {
      document.getElementById('side_bar').innerHTML += "bounds[" + searchIndex + "] returns 0 results<br>&nbsp;status=" + status + "<br>";
    }
    if (status != google.maps.places.PlacesServiceStatus.OVER_QUERY_LIMIT) {
      searchIndex++;
      if (searchIndex < boxes.length)
        findPlaces(searchIndex);
    } else { // delay 1 second and try again
      setTimeout("findPlaces(" + searchIndex + ")", 1000);
    }

  });
}

// Clear boxes currently on the map
function clearBoxes() {
  if (boxpolys != null) {
    for (var i = 0; i < boxpolys.length; i++) {
      boxpolys[i].setMap(null);
    }
  }
  boxpolys = null;
}

function createMarker(place) {
  var placeLoc = place.geometry.location;
  if (place.icon) {
    var image = new google.maps.MarkerImage(
      place.icon, new google.maps.Size(71, 71),
      new google.maps.Point(0, 0), new google.maps.Point(17, 34),
      new google.maps.Size(25, 25));
  } else var image = {
    url: "https://maps.gstatic.com/intl/en_us/mapfiles/markers2/measle.png",
    size: new google.maps.Size(7, 7),
    anchor: new google.maps.Point(3.5, 3.5)
  };

  var marker = new google.maps.Marker({
    map: map,
    icon: image,
    position: place.geometry.location
  });
  var request = {
    reference: place.reference
  };
  google.maps.event.addListener(marker, 'click', function() {
    service.getDetails(request, function(place, status) {
      if (status == google.maps.places.PlacesServiceStatus.OK) {
        var contentStr = '<h5>' + place.name + '</h5><p>' + place.formatted_address;
        if (!!place.formatted_phone_number) contentStr += '<br>' + place.formatted_phone_number;
        if (!!place.website) contentStr += '<br><a target="_blank" href="' + place.website + '">' + place.website + '</a>';
        contentStr += '<br>' + place.types + '</p>';
        infowindow.setContent(contentStr);
        infowindow.open(map, marker);
      } else {
        var contentStr = "<h5>No Result, status=" + status + "</h5>";
        infowindow.setContent(contentStr);
        infowindow.open(map, marker);
      }
    });

  });
  gmarkers.push(marker);
  if (!place.name) place.name = "result " + gmarkers.length;
  var side_bar_html = "<a href='javascript:google.maps.event.trigger(gmarkers[" + parseInt(gmarkers.length - 1) + "],\"click\");'>" + place.name + "</a><br>";
  document.getElementById('side_bar').innerHTML += side_bar_html;
}

function createGasStationMarker(coords,price,fill_liters,name,adress) {
  var placeLoc = coords;
  var image = {
    url: "https://maps.gstatic.com/intl/en_us/mapfiles/markers2/measle.png",
    size: new google.maps.Size(7, 7),
    anchor: new google.maps.Point(3.5, 3.5)
  };

  var marker = new google.maps.Marker({
    map: map,
    icon: image,
    position: coords
  });
  google.maps.event.addListener(marker, 'click', function() {
      var contentStr = '<h5>' + name + '</h5><p>Gas Price: ' + price + ' €<br>Fill: ' + fill_liters + ' l<br><br>' + adress + '</p>';
      infowindow.setContent(contentStr);
      infowindow.open(map, marker);
  });
  gmarkers.push(marker);

 describtion = '[' + gmarkers.length + "] " + name + " | " + price + " €";
  var side_bar_html = "<a href='javascript:google.maps.event.trigger(gmarkers[" + parseInt(gmarkers.length - 1) + "],\"click\");'>" + describtion + "</a><br>";
  document.getElementById('side_bar').innerHTML += side_bar_html;
}

google.maps.event.addDomListener(window, 'load', initialize);
