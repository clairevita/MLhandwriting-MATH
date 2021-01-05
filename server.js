const express = require("express");

const PORT = process.env.PORT || 8000;

const app = express();

app.use(express.static('/'));

//Here we are telling the host to use the Express framework.
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

var path = require("path");

app.get("/", function (req, res) {
  res.sendFile(path.join(__dirname, "index.html"));
});

function buildCreds() {
  require("dotenv").config();
}

buildCreds();
async function quickstart(myproject) {
  // Imports the Google Cloud client library
  const vision = require('@google-cloud/vision');
  // Creates a client
  console.log();
  const client = new vision.ImageAnnotatorClient({
    credentials: `${myproject}`
  });

  // Performs label detection on the image file
  const [result] = await client.labelDetection('./resources/dannyhembree2.jpg');
  const labels = result.labelAnnotations;
  console.log('Labels:');
  labels.forEach(label => console.log(label.description));
}


//This initiates the server.
app.listen(PORT, function () {
  console.log("App listening on PORT: " + PORT);
}); 
