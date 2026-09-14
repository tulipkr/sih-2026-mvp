import { API_BASE_URL, USE_PRECOMPUTED_FALLBACK, FALLBACK_RUN_RESULT_PATH } from "../config";

/*
  Loads the run result from Bhumika's backend.

  If the backend is unreachable, the frontend automatically
  loads public/demo_run_result.json.
*/

export async function loadRunResult(runId) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/results/${runId}`
    );

    /*
      A 404 means the backend responded but that run does not exist.
      We should not silently pretend it succeeded.
    */
    if (response.status === 404) {
      throw new Error(
        `Run ${runId} was not found on the backend.`
      );
    }

    if (!response.ok) {
      throw new Error(
        `Backend returned HTTP ${response.status}`
      );
    }

    const data = await response.json();

    return {
      data,
      fallbackUsed: false
    };

  } catch (error) {

    console.warn(
      "Live API unavailable. Trying bundled demo result.",
      error
    );

    if (!USE_PRECOMPUTED_FALLBACK) {
      throw error;
    }

    try {

      const fallbackResponse = await fetch(
        FALLBACK_RUN_RESULT_PATH
      );

      if (!fallbackResponse.ok) {
        throw new Error(
          "Bundled demo_run_result.json could not be loaded."
        );
      }

      const fallbackData =
        await fallbackResponse.json();

      return {
        data: fallbackData,
        fallbackUsed: true
      };

    } catch (fallbackError) {

      throw new Error(
        `Live API failed and fallback failed: ${fallbackError.message}`
      );

    }
  }
}