$(document).ready(() => {
    $("#produceBill").on("click", () => {
      const brand = $("#brand").val()
      const productTypes = $('input[name="productType"]:checked')
        .map(function () {
          return $(this).val()
        })
        .get()
      const quantity = $("#quantity").val()
  
      if (!brand || productTypes.length === 0 || !quantity) {
        alert("Please fill in all fields")
        return
      }
  
      const bill = generateBill(brand, productTypes, quantity)
      displayBill(bill)
  
      // Display alert with total amount
      alert(`Total amount: $${bill.total}`)
    })
  })
  
  function generateBill(brand, productTypes, quantity) {
    const prices = {
      HP: { Mobile: 300, Laptop: 800 },
      Nokia: { Mobile: 250, Laptop: 700 },
      Samsung: { Mobile: 400, Laptop: 900 },
      Motorola: { Mobile: 200, Laptop: 600 },
      Apple: { Mobile: 800, Laptop: 1500 },
    }
  
    let total = 0
    const items = []
  
    productTypes.forEach((type) => {
      const price = prices[brand][type]
      const itemTotal = price * quantity
      total += itemTotal
      items.push(`${brand} ${type}: $${price} x ${quantity} = $${itemTotal}`)
    })
  
    return {
      items: items,
      total: total,
    }
  }
  
  function displayBill(bill) {
    const billHtml = `
          <h2>Your Bill</h2>
          <ul>
              ${bill.items.map((item) => `<li>${item}</li>`).join("")}
          </ul>
          <p><strong>Total: $${bill.total}</strong></p>
      `
  
    $("#billOutput").html(billHtml)
  }
  
  