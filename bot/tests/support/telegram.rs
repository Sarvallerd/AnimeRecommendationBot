use bot::{
    dialogue::{callback::TokenSource, context::AppContext},
    handlers,
};
use serde_json::json;
use std::{
    io::{Read, Write},
    net::TcpListener,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};
use teloxide::{
    prelude::*,
    types::{CallbackQuery, Message, Update, UpdateId, UpdateKind},
};

#[allow(dead_code)]
pub struct FakeTelegram {
    url: String,
    requests: Arc<Mutex<Vec<(String, String)>>>,
    pub fail_send: Arc<AtomicBool>,
    pub fail_ack: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl FakeTelegram {
    pub fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let fail_send = Arc::new(AtomicBool::new(false));
        let fail_ack = Arc::new(AtomicBool::new(false));
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let seen = requests.clone();
        let failure = fail_send.clone();
        let ack_failure = fail_ack.clone();
        let thread = thread::spawn(move || {
            let mut next_id = 100;
            while !stopping.load(Ordering::SeqCst) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(_) => break,
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut bytes = Vec::new();
                let mut chunk = [0u8; 4096];
                while let Ok(n) = stream.read(&mut chunk) {
                    if n == 0 {
                        break;
                    }
                    bytes.extend_from_slice(&chunk[..n]);
                    if let Some(start) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&bytes[..start]);
                        let len = head
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .and_then(|value| value.trim().parse::<usize>().ok())
                            })
                            .unwrap_or(0);
                        if bytes.len() >= start + 4 + len {
                            break;
                        }
                    }
                }
                let request = String::from_utf8_lossy(&bytes);
                let path = request.split_whitespace().nth(1).unwrap_or("").to_owned();
                let body = request
                    .split_once("\r\n\r\n")
                    .map(|(_, body)| body.to_owned())
                    .unwrap_or_default();
                let fields = decode_fields(&body);
                let chat_id = fields
                    .get("chat_id")
                    .and_then(|v| v.parse::<i64>().ok())
                    .unwrap_or(1);
                seen.lock().unwrap().push((path.clone(), body));
                let response = if path.ends_with("/AnswerCallbackQuery") {
                    if ack_failure.swap(false, Ordering::SeqCst) {
                        json!({"ok":false,"error_code":500,"description":"failed"})
                    } else {
                        json!({"ok":true,"result":true})
                    }
                } else if failure.swap(false, Ordering::SeqCst) {
                    json!({"ok":false,"error_code":500,"description":"failed"})
                } else {
                    next_id += 1;
                    json!({"ok":true,"result":{"message_id":next_id,"date":1,"chat":{"id":chat_id,"type":"private","first_name":"Test"},"text":"ok"}})
                };
                let body = response.to_string();
                let wire=format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",body.len(),body);
                stream.write_all(wire.as_bytes()).unwrap();
            }
        });
        Self {
            url,
            requests,
            fail_send,
            fail_ack,
            stop,
            thread: Some(thread),
        }
    }
    #[allow(dead_code)]
    pub fn decoded(&self) -> Vec<(String, std::collections::HashMap<String, String>)> {
        self.snapshot()
            .into_iter()
            .map(|(path, body)| {
                let fields = decode_fields(&body);
                (path, fields)
            })
            .collect()
    }
    pub fn bot(&self) -> Bot {
        Bot::new("123:test").set_api_url(self.url.parse().unwrap())
    }
    pub fn snapshot(&self) -> Vec<(String, String)> {
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for FakeTelegram {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}
#[allow(dead_code)]
pub struct FixedTokens(pub AtomicUsize);
impl TokenSource for FixedTokens {
    fn token(&self) -> Result<String, getrandom::Error> {
        Ok(format!(
            "a1:{:032x}",
            self.0.fetch_add(1, Ordering::SeqCst) + 1
        ))
    }
}

pub fn message(id: u32, chat: i64, user: u64, text: &str) -> Update {
    let ty = if chat > 0 { "private" } else { "group" };
    let value = json!({"update_id":id,"message":{"message_id":id,"date":1,
        "chat":{"id":chat,"type":ty,"first_name":"Test","title":"Group"},
        "from":{"id":user,"is_bot":false,"first_name":"Тест"},"text":text}});
    Update {
        id: UpdateId(id),
        kind: UpdateKind::Message(
            serde_json::from_value::<Message>(value["message"].clone()).unwrap(),
        ),
    }
}
pub fn callback(id: u32, chat: i64, user: u64, message_id: i32, data: Option<&str>) -> Update {
    let value = json!({"update_id":id,"callback_query":{"id":format!("q{id}"),
        "from":{"id":user,"is_bot":false,"first_name":"Тест"},"chat_instance":"x",
        "message":{"message_id":message_id,"date":1,"chat":{"id":chat,"type":"private","first_name":"Test"}},
        "data":data}});
    Update {
        id: UpdateId(id),
        kind: UpdateKind::CallbackQuery(
            serde_json::from_value::<CallbackQuery>(value["callback_query"].clone()).unwrap(),
        ),
    }
}
pub async fn dispatch(bot: &Bot, ctx: &Arc<AppContext>, update: Update) {
    let result = handlers::schema()
        .dispatch(dptree::deps![bot.clone(), update, ctx.clone()])
        .await;
    assert!(
        matches!(result, std::ops::ControlFlow::Break(Ok(()))),
        "{result:?}"
    );
}

fn decode_fields(body: &str) -> std::collections::HashMap<String, String> {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str(body) {
        map.into_iter()
            .map(|(key, value)| {
                let text = value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string());
                (key, text)
            })
            .collect()
    } else {
        serde_urlencoded::from_str(body).unwrap_or_default()
    }
}
