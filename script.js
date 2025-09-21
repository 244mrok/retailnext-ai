// ユーザー名を保持する変数
let currentUser = null;

document.addEventListener('DOMContentLoaded', function () {
  const header = document.getElementById('header');
  const hamburgerBtn = document.getElementById('hamburger-btn');
  const mobileMenu = document.getElementById('mobile-menu');


  // スクロール時のヘッダー背景変更
  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  });

  // ハンバーガーメニューの開閉
  hamburgerBtn.addEventListener('click', () => {
    mobileMenu.classList.toggle('hidden');

    // アイコンをメニューから閉じるへ切り替え
    const icon = hamburgerBtn.querySelector('i');
    if (mobileMenu.classList.contains('hidden')) {
      icon.setAttribute('data-lucide', 'menu');
    } else {
      icon.setAttribute('data-lucide', 'x');
    }
    lucide.createIcons(); // アイコンを再描画
  });

  /*   const form = document.getElementById('chatbot-form');
     const input = document.getElementById('chatbot-input');
     const messages = document.getElementById('chatbot-messages');
     form.addEventListener('submit', function(e) {
       e.preventDefault();
       const userMsg = input.value.trim();
       if (!userMsg) return;
       messages.innerHTML += `<div style="margin-bottom:8px;"><b>あなた:</b> ${userMsg}</div>`;
       input.value = '';
       // ダミー応答（API連携部分をここに実装）
       setTimeout(() => {
         messages.innerHTML += `<div style="margin-bottom:8px;"><b>Bot:</b> 申し訳ありません、現在はデモ応答のみです。</div>`;
         messages.scrollTop = messages.scrollHeight;
       }, 600);
      messages.scrollTop = messages.scrollHeight;
     });
  */
  const form = document.getElementById('chatbot-form');
  const input = document.getElementById('chatbot-input');
  const messages = document.getElementById('chatbot-messages');
  const imageInput = document.getElementById('chatbot-image');

  imageInput.addEventListener('change', function () {
    const imageFile = imageInput.files[0];
    if (imageFile) {
      const reader = new FileReader();
      reader.onload = function (e) {
        messages.innerHTML += `<div style="margin-bottom:8px;"><b>You:</b> Image Uploaded<br>
        <img src="${e.target.result}" alt="Uploaded Image" style="max-width:120px; max-height:120px; border-radius:8px; margin-top:4px;"/>`;
        messages.scrollTop = messages.scrollHeight;
      };
      reader.readAsDataURL(imageFile);
    }
  });

  form.addEventListener('submit', async function (e) {
    e.preventDefault();
    const userMsg = input.value.trim();
    const imageFile = imageInput.files[0];

    if (userMsg) {
      // currentUserがセットされていれば名前を表示
      const userLabel = currentUser ? `${currentUser}` : "You";
      messages.innerHTML += `<div style="margin-bottom:8px;"><b>${userLabel}:</b> ${userMsg}</div>`;
      input.value = '';
      const formData = new FormData();
      formData.append('email', currentUser ? currentUser : 'unknown');
      formData.append('question', userMsg);
      const res = await fetch('http://localhost:8000/chatbot_answer', {
        method: 'POST',
        body: formData
      });
      const html = await res.text();
      document.getElementById('chatbot-messages').innerHTML += html;
    }

    if (imageFile) {
      //messages.innerHTML += `<div style="margin-bottom:8px;"><b>あなた:</b> 画像をアップロードしました。</div>`;
      // API呼び出し
      const formData = new FormData();
      formData.append('file', imageFile);

      //const response = await fetch('http://localhost:8000/recommend', {
      const response = await fetch('http://localhost:8000/recommendwithselected', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        messages.innerHTML += `<div style="margin-bottom:8px;"><b>Bot:</b> API Error: ${response.status}</div>`;
        return;
      }
      const data = await response.json();
      console.log("API Response:", data); // デバッグ用ログ

      // 推薦結果を表示
      //if (data.recommendations && data.recommendations.length > 0) {
      if (data.recommendations) {
        messages.innerHTML += `<div style="margin-bottom:8px;"><b>Bot:</b> Recommend Items:</div>`;
        //data.recommendations.forEach(item => {
        // messages.innerHTML += `<div style="margin-bottom:4px;">・${item.name} (${item.category}, ${item.gender})</div>`;
        //   messages.innerHTML += item.html;
        //});
        messages.innerHTML += data.recommendations.html;
      } else {
        messages.innerHTML += `<div style="margin-bottom:8px;"><b>Bot:</b> Recommend Items not found.</div>`;
      }
      messages.scrollTop = messages.scrollHeight;
      imageInput.value = '';
    }
  });

  // ログインモーダル表示（例: ユーザーアイコン押下で表示）
  const userBtn = document.getElementById('user-btn');
  if (userBtn) {
    userBtn.addEventListener('click', () => {
      document.getElementById('login-modal').style.display = 'flex';
    });
  }

  // ログイン処理（Mock）
  document.getElementById('login-btn').addEventListener('click', async function () {
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    if (username && password) {
      currentUser = username; // ユーザー名を保持
      document.getElementById('login-modal').style.display = 'none';
      // チャット欄にログインユーザー名を表示
      const messages = document.getElementById('chatbot-messages');
      messages.innerHTML += `<div style="margin-bottom:8px; color:#374151;"> ${currentUser} logged in</div>`;
      
      // ここでバナー画像をAIで選択して切り替え(非同期)
      fetchBestBanner(currentUser);
      // おすすめ商品を取得して表示(非同期)
      fetchRecommendItems(currentUser);

      alert(`Welcome, ${username}！（Mock Login）`);

    } else {
      alert('Please enter your username and password');
    }
  });

  // jsではCurrent User = email
  async function fetchBestBanner(email) {
    const formData = new FormData();
    formData.append("email", email);

    try {
      const res = await fetch("http://localhost:8000/select_banner", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.banner_path) {
        // バナー画像を表示
        const heroBanner = document.getElementById("hero-banner");
        if (heroBanner) {
          heroBanner.style.backgroundImage = `url('${data.banner_path}')`;
        }
        // 理由を表示（任意）
        if (data.reason) {
          console.log("Selected Banner Reason:", data.reason);
        }
      } else if (data.error) {
        alert("バナー選択エラー: " + data.error);
      }
    } catch (err) {
      alert("通信エラー: " + err);
    }
  }

  async function fetchRecommendItems(email) {
    const formData = new FormData();
    formData.append("email", email);
    try {
      const res = await fetch("http://localhost:8000/recommend_items", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (data.items) {
        const grid = document.querySelector('.grid.grid-cols-2.md\\:grid-cols-4');
        grid.innerHTML = ""; // 既存カードを消す
        data.items.forEach(item => {
         grid.innerHTML += `
            <div class="group">
              <div class="aspect-[3/4] bg-gray-100 rounded-lg overflow-hidden mb-2">
                <img src="${item.img}" alt="${item.name}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300">
             </div>
             <h3 class="font-bold">${item.name}</h3>
             <p class="text-gray-600">${"$" + item.price}</p>
           </div>
          `;
       });
      } else if (data.error) {
        alert("おすすめ取得エラー: " + data.error);
      }
    } catch (err) {
      alert("通信エラー: " + err);
    }
  }



  // モーダル外クリックで閉じる
  document.getElementById('login-modal').addEventListener('click', function (e) {
    if (e.target === this) this.style.display = 'none';
  });

  // チャットボット開閉ボタン
  const chatbotContainer = document.getElementById('chatbot-container');
  const chatbotToggleBtn = document.getElementById('chatbot-toggle-btn');
  const chatbotCloseBtn = document.getElementById('chatbot-close-btn');

  chatbotToggleBtn.addEventListener('click', function () {
    chatbotContainer.style.display = 'flex';
    chatbotToggleBtn.style.display = 'none';
  });

  chatbotCloseBtn.addEventListener('click', function () {
    chatbotContainer.style.display = 'none';
    chatbotToggleBtn.style.display = 'block';
  });

});





