const express = require("express");

const PORT = process.env.PORT || 8000;

const app = express();

app.use(express.static('/'));

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

var path = require("path");

app.get("/", function (req, res) {
  res.sendFile(path.join(__dirname, "index.html"));
});
quickstart();
async function quickstart() {
  // Imports the Google Cloud client library
  const vision = require('@google-cloud/vision');

  // Creates a client
  const client = new vision.ImageAnnotatorClient({
    keyFilename: "./node_modules/google-gax/My Project-528abf0f6cfb.json"
  });
  const [result] = await client.documentTextDetection("./resources/5.png");
  const fullTextAnnotation = result.fullTextAnnotation;
  console.log(`Full text: ${fullTextAnnotation.text}`);
  fullTextAnnotation.pages.forEach(page => {
    page.blocks.forEach(block => {
      console.log(`Block confidence: ${block.confidence}`);
      block.paragraphs.forEach(paragraph => {
        console.log(`Paragraph confidence: ${paragraph.confidence}`);
        paragraph.words.forEach(word => {
          const wordText = word.symbols.map(s => s.text).join('');
          console.log(`Word text: ${wordText}`);
          console.log(`Word confidence: ${word.confidence}`);
        });
      });
    });
  });
}
//This initiates the server.
app.listen(PORT, function () {
  console.log("App listening on PORT: " + PORT);
}); 
