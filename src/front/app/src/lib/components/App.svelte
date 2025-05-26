<script lang="ts">
  import TextInput from "./TextInput.svelte";
  import BarChart from "./BarChart.svelte";
  import { fly , fade } from "svelte/transition";

  let models = [
    { label: "GPT-4o", value: "gpt-4o" },
    { label: "Llama 3.1", value: "llama-3.1" },
    { label: "Logistic Regression", value: "logistic_regression" },
    { label: "Naive Bayes", value: "naive_bayes" },
    { label: "SVM", value: "svm" },
    { label: "GRU", value: "gru" },
    { label: "BERT", value: "bert-base-uncased" },
    { label: "RoBERTa", value: "roberta-base" },
    { label: "XLM-RoBERTa", value: "xlm-roberta-base" },
  ];

  let inputText = "";
  let predictions: Record<
    string,
    { etiqueta_aita: string; razonamiento: string }
  > = {};
  let loading = false;

  async function predict() {
    if (!inputText) return;
    predictions = {};
    loading = true;

    const fastModels = models.filter((m) =>
      ["logistic_regression", "naive_bayes", "svm", "gru"].includes(m.value),
    );
    const slowModels = models.filter((m) =>
      ["gpt-4o", "llama-3.1", "bert-base-uncased", "roberta-base", "xlm-roberta-base"].includes(m.value),
    );

    const fetchPredictions = async (
      modelList: { label: string; value: string }[],
    ) => {
      const results: Record<
        string,
        { etiqueta_aita: string; razonamiento: string }
      > = {};

      await Promise.all(
        modelList.map(async (model) => {
          try {
            const res = await fetch(
              `http://localhost:8000/predict/${model.value}`,
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: inputText }),
              },
            );

            const data = await res.json();
            results[model.label] = data.result;
          } catch (err) {
            results[model.label] = {
              etiqueta_aita: "Error",
              razonamiento: "Failed to fetch.",
            };
          }
        }),
      );

      return results;
    };

    // Fetch both sets
    const [fastResults, slowResults] = await Promise.all([
      fetchPredictions(fastModels),
      fetchPredictions(slowModels),
    ]);

    // Now update predictions just once
    predictions = { ...fastResults, ...slowResults };

    loading = false;
  }

  function voteDistribution() {
    const count: Record<string, number> = {};
    for (const result of Object.values(predictions)) {
      const vote = result.etiqueta_aita;
      count[vote] = (count[vote] || 0) + 1;
    }
    return count;
  }
</script>

<div class="container">
  <h1>r/AITA Predicción de veredicto utilizando machine learning</h1>

  <TextInput bind:value={inputText} placeholder="Pega aquí el post de r/AITA..." />

  <button
    class="upload-button"
    on:click={predict}
    disabled={!inputText || loading}
  >
    {#if loading}Procesando{:else}Predecir{/if}
  </button>

  {#if Object.keys(predictions).length > 0}
    <div class="model-results" in:fade={{ duration: 400 }} out:fade={{ duration: 400 }}>
      {#each models as model}
        <div class="model-card" in:fly={{ y: 20, duration: 300 }} out:fly={{ y: -20, duration: 300 }}>
          {#if predictions[model.label]}
          <h3>
            {model.label}: <span style="color: #007bff; font-style: italic;">{predictions[model.label].etiqueta_aita}</span>
          </h3>
            <div>
              {#if model.label === "GPT-4o" || model.label === "Llama 3.1"}
                <p>
                  <strong>Razonamiento:</strong>
                  {predictions[model.label].razonamiento}
                </p>
              {/if}
            </div>
          {:else}
            <p class="loading">Cargando...</p>
          {/if}
        </div>
      {/each}
    </div>

    <BarChart data={voteDistribution()} />
  {/if}
</div>

<style>
  .container {
    max-width: 900px;
    margin: 2rem auto;
    padding: 1rem;
    font-family: system-ui, sans-serif;
    text-align: center;
  }

  .upload-button {
    padding: 0.75rem 1.5rem;
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    margin: 1rem 0;
    cursor: pointer;
  }

  .upload-button:disabled {
    background-color: #aaa;
    cursor: not-allowed;
  }

  .model-results {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1.5rem;
  }

  .model-card {
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 1rem;
    background: #f9f9f9;
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
  }

  .loading {
    color: #888;
    font-style: italic;
    animation: pulse 1.2s infinite;
  }

  @keyframes pulse {
    0% {
      opacity: 0.4;
    }
    50% {
      opacity: 1;
    }
    100% {
      opacity: 0.4;
    }
  }

</style>
