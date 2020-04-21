// version with nlp-compromise instead of RiTa for tokenization
// this is the first attempt with successful outcome!

var model; // the NN model

// two main input and output arrays for the training of the neural network:
var training = [];
var intent_tags = []; // the tag array for each phrase (all tags found including duplicates)

var unique_words = []; // all unique words found in user patterns (no duplicates)
var labels = []; // the tags (single tags - no duplicates: the list of unique tags)

// user input and response textbox:
var user_input;
var bot_response;
var submit_button;

function preload() {
	// Load the training set and create the bag of words:
    prepare_training_set();
}


function setup() {

	// user input and response textbox:
	user_input = select('#input');
	bot_response = select('#response');
	submit_button = select('#submit');

    // initialize the neural network:
	NN_initialize(unique_words.length, labels.length);

	// when submit button clicked:
	submit_button.mousePressed(() => {
		// get input phrase from user:
    	let phrase = str(user_input.value());

    	// process the response and display on the text box:
    	proccess_phrase(phrase);

    }); // end of submit button callback

} // end of setup()


// load the intent tags JSON file which will serve a the training set and create the bag of words:
function prepare_training_set() {
	// load the training JSON file:
	loadJSON("intents.json", call);

	// main callback after JSON file is loaded:
	function call(data) {
    	console.log("Intents file loaded successfully");

    	// Extract features of the JSON file (needs to be done inside the callback after the file is loaded):
    	// the words array to store words of each pattern:
    	var all_words = []; // all words found in the user patterns (including duplicates)
    	var pattern_word_list = []; // array of sublists of all words for each pattern

    	// loop through the JSON data to extract the user pattern words:
		for (var i = 0; i < data.intents.length; i++) { // i is each intent row

			for (var j = 0; j < data.intents[i].patterns.length; j++) { // j is each pattern in patterns

				// convert string to nlp object (removes punctuation also):
				var string = nlp(data.intents[i].patterns[j]);

				// get the tokenized word array (split into words) from the pattern phrase:
				var arr = string.termList(); // arr is an array of objects
				
				var pattern = [];
				for (var r = 0; r < arr.length; r ++) {
					pattern[r] = arr[r].text;
				}
				
				// append to the all_words array as array. This array is used only to create the unique words:
				all_words = all_words.concat(pattern);
				
				// add the pattern word list as item to pattern_word_list array (array of arrays):
				pattern_word_list.push(pattern); 

				// add intent tag to tags array (same size as the patten words array - they repeat):
				intent_tags.push(data.intents[i].tag);

				// create array of single (non-repeating) labels (tags):
				if ( labels.includes(data.intents[i].tag) == false) {
					labels.push(data.intents[i].tag);
				}
			}
		}
		
		// now create a list of unique word elements (like a set in python):
		let word_set = new Set(all_words);
		// convert back to array:
		unique_words = Array.from(word_set);
		// and sort alphabetically:
		unique_words = unique_words.sort();

		// sort the labels:
		labels = labels.sort();

		// the bag of words encodes the pattern phrases: it creates an array of all the words appearing in the 
		//user patterns. Each vector length will be equal to the amount of unique words in the intents space.
		// It is essentially a binary histogram of word appearances (1 if exists, 0 if not). 
		// create the bag of words in the training array. For each phrase in all pattern phrases:
		for (var phrase = 0; phrase < pattern_word_list.length; phrase ++) {
			// initialize each phrase bag:
			var bag = [];

			// for each word in the array of unique words:
			for (var uw = 0; uw < unique_words.length; uw ++) {

				// check if this word is found in the phrase:
				if (pattern_word_list[phrase].includes(unique_words[uw])) {
					// put 1 in the bag of the phrase for this word:
					bag.push(1);
					//console.log(pattern_word_list[phrase] + " includes word " + unique_words[uw]);
				}
				else {
					// put 0 in the bag of the phrase for this word. Note: we put just 1 if the word exists,
					// 0 if it doesn't. We don't care about the frequency (how many times it exists). 
					bag.push(0);
				}
			}
			// now insert the phrase bag to the training array (this is the total bag of words):
			training.push(bag);
		}
	}; // end of JSON callback
}


function NN_initialize(input_size, output_size) {

	// specify options for initializing the NN corresponding to the size of input and output array:
	let options = {
		inputs: input_size, // how many different inputs
		outputs: output_size, // how many different outputs
		task: 'classification',
		debug: true
	}
	// initialize the NN:
	model = ml5.neuralNetwork(options);

	console.log("NN model initialized");
}


function keyPressed() {
	// get the training data: for each phrase insert its bag of words as input and the tag position as output:
	if (key == 'T') {

		for (var i = 0; i < training.length; i++) { // for all phrases:
			model.addData(training[i], [intent_tags[i]]); // the output should be in array format!
		}

		// start the training process:
		console.log("Started training");

		// normalize the data:
		model.normalizeData();

		// specify training options:
		const trainingOptions = {
			epochs: 200
		}

		// now train the model with the data given before and the training options:
		model.train(trainingOptions, whileTraining, finishedTraining);

		// specify the callbacks:
		function whileTraining(epoch, loss) {
			// monitor the training progress
			//console.log(epoch, loss);
		}

		function finishedTraining() {
			console.log("Finished training");
		}
	}

	// save the data:
	if (key == 'S') {
		model.saveData("model-data");
	}
}


// create word list of input phrase and choose random answer from list of responses
function proccess_phrase(phrase) {
	// convert string to nlp object:
	var string = nlp(phrase);

	// get the tokenized word array (split into words) from the pattern phrase:
	var arr = string.termList(); // pattern is an array of strings

	var pattern = [];
	for (var r = 0; r < arr.length; r ++) {
		pattern[r] = arr[r].text;
	}

	// create the bag of words of the input phrase based on the unique words array created from the training set:
	let bag = [];

	// for each word in the array of unique words:
	for (var uw = 0; uw < unique_words.length; uw ++) {

		// check if this word is found in the phrase:
		if ( pattern.includes(unique_words[uw]) ) {
			// put 1 in the bag of the phrase for this word:
			bag.push(1);
		}
		else {
			// put 0 in the bag of the phrase for this word:
			bag.push(0);
		}
	}

	// now classify the phrase tag:
	model.classify(bag, gotResults);
}

function gotResults(error, results) {
	if (error) {
		console.log(error);
	}
	console.log(results);

	// get the most probable intent tag (place 0) from the result:
	response = results[0].label;	

	// display the response in the text field:
	bot_response.value(response);
  		
	// now map the result to a random phrase from the intents list
	
}




