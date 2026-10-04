const getData = async () => {
  console.log("Clickd")
  const url = 'http://localhost:8000'
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { "Content-Type": "application/json"},
      body: JSON.stringify({'message': 'what is vllm?', 'k': 5})
    })
    if (!response.ok) {
      console.log(await response.json())
      throw new Error(`Response Status: ${response.status}`)
    }
    const result = await response.json()
    console.log(result)
  } catch (error) {
    console.error(error)
  }
}
