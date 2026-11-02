let experiment = null;
let trials = [];

let currentTrialIndex = 0;
let trialStartedAt = null;
let timeoutHandle = null;

let responses = [];


const startScreen =
  document.getElementById("start-screen");

const taskScreen =
  document.getElementById("task-screen");

const endScreen =
  document.getElementById("end-screen");

const startButton =
  document.getElementById("start-button");

const leftButton =
  document.getElementById("left-choice");

const rightButton =
  document.getElementById("right-choice");

const leftImage =
  document.getElementById("left-image");

const rightImage =
  document.getElementById("right-image");

const progress =
  document.getElementById("progress");

const timeoutMessage =
  document.getElementById("timeout-message");


function getParticipantId() {
  const params = new URLSearchParams(
    window.location.search
  );

  return (
    params.get("participant_id")
    || params.get("PROLIFIC_PID")
    || "LOCAL_TEST"
  );
}


function shuffle(array) {
  const result = [...array];

  for (
    let index = result.length - 1;
    index > 0;
    index -= 1
  ) {
    const randomIndex = Math.floor(
      Math.random() * (index + 1)
    );

    [
      result[index],
      result[randomIndex],
    ] = [
      result[randomIndex],
      result[index],
    ];
  }

  return result;
}


async function loadExperiment() {
  const response = await fetch(
    "experiment.json"
  );

  if (!response.ok) {
    throw new Error(
      "Unable to load experiment.json"
    );
  }

  experiment = await response.json();

  trials = [...experiment.trials];

  if (experiment.randomize_trials) {
    trials = shuffle(trials);
  }
}


function showScreen(screen) {
  startScreen.classList.add("hidden");
  taskScreen.classList.add("hidden");
  endScreen.classList.add("hidden");

  screen.classList.remove("hidden");
}


function clearTrialTimeout() {
  if (timeoutHandle !== null) {
    clearTimeout(timeoutHandle);
    timeoutHandle = null;
  }
}


function showTrial() {
  clearTrialTimeout();

  timeoutMessage.classList.add(
    "hidden"
  );

  if (currentTrialIndex >= trials.length) {
    finishExperiment();
    return;
  }

  const trial =
    trials[currentTrialIndex];

  progress.textContent =
    `Essai ${currentTrialIndex + 1} / ${trials.length}`;

  leftImage.src =
    trial.image_left;

  rightImage.src =
    trial.image_right;

  trialStartedAt =
    performance.now();

  if (
    experiment.response_timeout_ms
    !== null
  ) {
    timeoutHandle = setTimeout(
      handleTimeout,
      experiment.response_timeout_ms
    );
  }
}


function saveResponse(
  trial,
  selectedModel,
  responseTimeMs
) {
  responses.push({
    participant_id:
      getParticipantId(),

    trial_index:
      currentTrialIndex + 1,

    pair_id:
      trial.pair_id,

    model_left:
      trial.model_left,

    model_right:
      trial.model_right,

    selected_model:
      selectedModel,

    response_time_ms:
      responseTimeMs,
  });
}


function handleChoice(side) {
  clearTrialTimeout();

  const trial =
    trials[currentTrialIndex];

  const responseTimeMs =
    Math.round(
      performance.now()
      - trialStartedAt
    );

  const selectedModel =
    side === "left"
      ? trial.model_left
      : trial.model_right;

  saveResponse(
    trial,
    selectedModel,
    responseTimeMs
  );

  currentTrialIndex += 1;

  showTrial();
}


function handleTimeout() {
  const trial =
    trials[currentTrialIndex];

  timeoutMessage.classList.remove(
    "hidden"
  );

  saveResponse(
    trial,
    null,
    experiment.response_timeout_ms
  );

  currentTrialIndex += 1;

  setTimeout(
    showTrial,
    800
  );
}


function downloadLocalResults() {
  const data = JSON.stringify(
    responses,
    null,
    2
  );

  const blob = new Blob(
    [data],
    {
      type: "application/json",
    }
  );

  const url =
    URL.createObjectURL(blob);

  const link =
    document.createElement("a");

  link.href = url;

  link.download =
    `responses_${getParticipantId()}.json`;

  link.click();

  URL.revokeObjectURL(url);
}


function finishExperiment() {
  clearTrialTimeout();

  showScreen(
    endScreen
  );

  console.log(
    "Participant responses:",
    responses
  );

  /*
   * Temporary local-development behaviour.
   *
   * JATOS submission will replace this
   * in the next step.
   */
  downloadLocalResults();
}


startButton.addEventListener(
  "click",
  () => {
    currentTrialIndex = 0;
    responses = [];

    showScreen(
      taskScreen
    );

    showTrial();
  }
);


leftButton.addEventListener(
  "click",
  () => {
    handleChoice(
      "left"
    );
  }
);


rightButton.addEventListener(
  "click",
  () => {
    handleChoice(
      "right"
    );
  }
);


async function initialise() {
  try {
    await loadExperiment();

    showScreen(
      startScreen
    );

  } catch (error) {
    console.error(
      error
    );

    document.body.innerHTML =
      "<p>Impossible de charger l'expérience.</p>";
  }
}


initialise();