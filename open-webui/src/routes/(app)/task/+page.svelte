<script>
  import { onMount } from "svelte";
  import axios from "axios";

  /**
	 * @type {any[]}
	 */
  let documents = [];
  let taskId = "";
  let taskStatus = "";
  let showModal = false;

  let modelConfig = {
    modelName: "BAAI/bge-small-en-v1.5",
    modelBaseUrl: "http://127.0.0.1:8081",
    modelEmbedBatchSize: 10,
    modelTimeout: 60.0,
    docHost: "localhost",
    docPort: 6379,
    docCollectionName: "document_store",
    cacheHost: "127.0.0.1",
    cachePort: 6379,
    cacheCollectionName: "cache",
    milvusURI: "http://localhost:19530",
    milvusDimension: 1024,
    milvusOverwrite: false,
    milvusCollectionName: "app_milvus_db",
  };

  // Fetch tasks on component mount
  async function fetchTasks() {
    try {
      const response = await axios.get("http://localhost:8000/tasks");
      documents = Object.entries(response.data).map(([id, task]) => ({
        id,
        description: task.description || "No description available",
      }));
    } catch (error) {
      console.error("Error fetching tasks:", error);
    }
  }

   // Create a new task
  async function createTask() {
    try {
      const response = await axios.post("http://localhost:8000/read", modelConfig);
      taskId = response.data.LoadTaskID;
      alert(`Task created with ID: ${taskId}`);
    } catch (error) {
      console.error("Error creating task:", error);
    }
  }

  // Fetch status of a specific task
  /**
	 * @param {any} id
	 */
  async function getTaskStatus(id) {
    try {
      const response = await axios.get(`http://localhost:8000/tasks/${id}`);
      taskStatus = response.data.status;
      showModal = true; // Show the modal
    } catch (error) {
      console.error("Error fetching task status:", error);
    }
  }

  // Close the modal
  function closeModal() {
    showModal = false;
  }

  onMount(fetchTasks);
</script>

<main>
  <section>
    <!-- Add a scrollable container for the table -->
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Description</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {#each documents as task}
            <tr>
              <td>{task.id}</td>
              <td>{task.description}</td>
              <td>
                <button on:click={() => getTaskStatus(task.id)}>Check Status</button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <!-- Modal for Task Status -->
  {#if showModal}
    <div class="modal">
      <div class="modal-content">
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <span class="close" on:click={closeModal}>&times;</span>
        <h2>Task Status</h2>
        <p>{taskStatus}</p>
      </div>
    </div>
  {/if}
</main>

<style>
  /* Ensure full-screen scrollability */
  body {
    margin: 0;
    padding: 0;
    font-family: Roboto;
    overflow-y: auto; /* Enables vertical scrolling */
    background-color: #f8f9fa;
  }

  main {
    padding: 2em;
    margin-bottom: 50px;
    margin-left: 200px;
    background-color: #f8f9fa;
    min-height: 100vh; /* Ensures the main section spans the viewport height */
  }

  h2 {
    color: #2d3436;
    margin-bottom: 1em;
  }

  /* Scrollable Table Container */
  .table-container {
    max-height: 660px; /* Adjust height as needed */
    overflow-y: auto; /* Enables vertical scrolling */
    border: 1px solid #ddd; /* Optional: Adds a border around the container */
    background-color: white;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1.5em;
  }

  th, td {
    padding: 1em;
    text-align: left;
    border-bottom: 1px solid #ddd; /* Adds borders to table rows */
  }

  th {
    background-color: black;
    color: white;
    position: sticky; /* Keeps the header visible during scrolling */
    top: 0; /* Sticky header positioning */
    z-index: 1; /* Ensures the header is above the table body */
  }

  tr:nth-child(even) {
    background-color: #e5e4e2;
  }

  tr:hover {
    background-color: #fafafa;
  }

  button {
    padding: 0.5em 1em;
    background-color: black;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.3s;
  }

  button:hover {
    background-color: #e5e4e2;
    color: black;
  }

  /* Modal Styles */
  .modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background-color: white;
    padding: 2em;
    border-radius: 8px;
    width: 50%;
    max-width: 500px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    position: relative;
  }

  .modal-content h2 {
    margin-bottom: 1em;
    color: #2d3436;
  }

  .modal-content p {
    font-size: 1.2em;
    color: #636e72;
  }

  .close {
    position: absolute;
    top: 1em;
    right: 1em;
    font-size: 1.5em;
    color: #636e72;
    cursor: pointer;
  }

  .close:hover {
    color: #d63031;
  }
</style>
