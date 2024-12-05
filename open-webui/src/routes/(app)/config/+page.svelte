<script>
  let task_data_management_collection = "task_data";
  let taskHost = "localhost";
  let taskPort = 6379;

  let modelName = "BAAI/bge-small-en-v1.5";
  let modelBaseUrl = "http://127.0.0.1:8081";
  let modelEmbedBatchSize = 10;

  let cacheHost = "127.0.0.1";
  let cachePort = 6379;
  let cacheCollectionName = "cache_collection";

  let milvusURI = "http://localhost:19530";
  let milvusDimension = 384;
  let milvusCollectionName = "app_milvus_db";

  let format = "web";
  let url_web = "";
  let url_html = "";
  let tag = "section";
  let ignore_no_id = true;
  let url_soup = "";
  let url_whole = "";
  let prefix = "";
  let max_depth = 3;
  let url_file = "";

  let responseData = "";
  let errorMessage = "";

  async function submitReadRequest() {
    try {
      const response = await fetch("http://127.0.0.1:8000/read", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          task_data_management_collection,
          taskHost,
          taskPort,
          modelName,
          modelBaseUrl,
          modelEmbedBatchSize,
          cacheHost,
          cachePort,
          cacheCollectionName,
          milvusURI,
          milvusDimension,
          milvusCollectionName,
          format,
          url_web,
          url_html,
          tag,
          ignore_no_id,
          url_soup,
          url_whole,
          prefix,
          max_depth,
          url_file,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Unknown error occurred");
      }

      const data = await response.json();
      responseData = data ? JSON.stringify(data, null, 2) : "No data returned from server.";
      errorMessage = "";
    } catch (error) {
      errorMessage = error.message;
      responseData = "";
    }
  }
</script>

<main>
  <section class="model-section">
    <h2>Model Configuration</h2>
    <div>
      <label>Model Name:</label>
      <input bind:value={modelName} type="text" />
    </div>
    <div>
      <label>Model Base URL:</label>
      <input bind:value={modelBaseUrl} type="text" />
    </div>
    <div>
      <label>Model Embed Batch Size:</label>
      <input bind:value={modelEmbedBatchSize} type="number" />
    </div>
  </section>

  <section class="document-section">
    <h2>Document Store Configuration</h2>
    <div>
      <label>Task Data Collection:</label>
      <input bind:value={task_data_management_collection} type="text" />
    </div>
    <div>
      <label>Task Host:</label>
      <input bind:value={taskHost} type="text" />
    </div>
    <div>
      <label>Task Port:</label>
      <input bind:value={taskPort} type="number" />
    </div>
  </section>

  <section class="vector-section">
    <h2>Vector Store Configuration</h2>
    <div>
      <label>Milvus URI:</label>
      <input bind:value={milvusURI} type="text" />
    </div>
    <div>
      <label>Milvus Dimension:</label>
      <input bind:value={milvusDimension} type="number" />
    </div>
    <div>
      <label>Milvus Collection Name:</label>
      <input bind:value={milvusCollectionName} type="text" />
    </div>
  </section>

  <section class="cache-section">
    <h2>Cache Configuration</h2>
    <div>
      <label>Cache Host:</label>
      <input bind:value={cacheHost} type="text" />
    </div>
    <div>
      <label>Cache Port:</label>
      <input bind:value={cachePort} type="number" />
    </div>
    <div>
      <label>Cache Collection Name:</label>
      <input bind:value={cacheCollectionName} type="text" />
    </div>
  </section>

  <section class="content-ingestion-section">
    <h2>Content Ingestion</h2>
    <div>
      <label for="format">Format:</label>
      <select bind:value={format}>
        <option value="dir">Directory Reader</option>
        <option value="web">Web Reader</option>
        <option value="html_tags">HTML Tags Reader</option>
        <option value="beautiful_soup">BeautifulSoup Reader</option>
        <option value="rss">RSS Reader</option>
        <option value="pdf">PDF Reader</option>
        <option value="docx">DOCX Reader</option>
        <option value="txt">TXT Reader</option>
        <option value="image">Image Reader</option>
        <option value="ipynb">IPYNB Reader</option>
        <option value="pptx">PPTX Reader</option>
        <option value="csv">CSV Reader</option>
        <option value="xml">XML Reader</option>
        <option value="whole_site">Whole Site Reader</option>
      </select>
    </div>

    {#if format === "web"}
      <div><label>URL:</label> <input bind:value={url_web} type="text" /></div>
    {/if}

    {#if format === "html_tags"}
      <div><label>URL:</label> <input bind:value={url_html} type="text" /></div>
      <div><label>Tag:</label> <input bind:value={tag} type="text" /></div>
      <div><label>Ignore No ID:</label> <input bind:checked={ignore_no_id} type="checkbox" /></div>
    {/if}

    {#if format === "beautiful_soup"}
      <div><label>URL:</label> <input bind:value={url_soup} type="text" /></div>
    {/if}

    {#if format === "whole_site"}
      <div><label>Base URL:</label> <input bind:value={url_whole} type="text" /></div>
      <div><label>Prefix:</label> <input bind:value={prefix} type="text" /></div>
      <div><label>Max Depth:</label> <input bind:value={max_depth} type="number" /></div>
    {/if}

    {#if format === "dir"}
      <div><label>File Path:</label> <input bind:value={url_file} type="text" /></div>
    {/if}

    {#if ["pdf", "docx", "txt", "image", "ipynb", "pptx", "csv", "xml"].includes(format)}
      <div><label>File Path:</label> <input bind:value={url_file} type="text" /></div>
    {/if}
  </section>

  <button on:click={submitReadRequest}>Submit</button>

  {#if responseData}
    <pre>{responseData}</pre>
  {/if}

  {#if errorMessage}
    <h2>Error:</h2>
    <p>{errorMessage}</p>
  {/if}
</main>

<style>
  main {
    font-family: Arial, sans-serif;
    margin: 20px auto;
    max-width: 800px;
    max-height: 90vh;
    overflow-y: auto;
    background-color: #e5e4e2;
    padding: 20px;
    border-radius: 10px;
  }

  section {
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 8px;
    background-color: #ffffff;
  }

  h2 { margin-bottom: 10px; }
  div { margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
  input, select {
    flex: 1; 
    margin-left: 10px; 
    padding: 5px; 
    border: 1px solid #ccc; 
    border-radius: 4px;
  }
  button {
    margin-top: 20px;
    padding: 10px 20px;
    background-color: #000;
    color: #fff;
    border: none;
    cursor: pointer;
    border-radius: 5px;
  }
  button:hover { opacity: 0.8; }
  pre {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 5px;
    overflow-x: auto;
    margin-top: 15px;
  }
</style>
