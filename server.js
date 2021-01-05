const express = require("express");

const PORT = process.env.PORT || 8000;

const app = express();

app.use(express.static('/'));

//Here we are telling the host to use the Express framework.
app.use(express.urlencoded({ extended: true }));
app.use(express.json());


var path = require("path");

app.get("/", function(req, res) {
    res.sendFile(path.join(__dirname, "index.html"));
  });

//This initiates the server.
app.listen(PORT, function () {
    console.log("App listening on PORT: " + PORT);
}); 
