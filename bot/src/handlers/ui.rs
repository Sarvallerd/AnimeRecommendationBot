//! Shared plain-text Telegram keyboard delivery.
use crate::dialogue::{callback::Action, state::Actor, storage::Session};
use std::error::Error;
use teloxide::{
    prelude::*,
    types::{InlineKeyboardButton, InlineKeyboardMarkup, Message},
};

pub(crate) fn bounded(text: &str, max_utf16: usize) -> String {
    let mut out = String::new();
    let mut units = 0;
    for original in text.chars() {
        let ch = if original.is_control() { ' ' } else { original };
        if units + ch.len_utf16() > max_utf16 {
            if max_utf16 > 0 {
                while units + 1 > max_utf16 {
                    if let Some(last) = out.pop() {
                        units -= last.len_utf16();
                    } else {
                        break;
                    }
                }
                out.push('…');
            }
            break;
        }
        out.push(ch);
        units += ch.len_utf16();
    }
    out
}

pub(crate) async fn send_keyboard(
    bot: &Bot,
    actor: Actor,
    session: &mut Session,
    text: String,
    rows: Vec<Vec<(String, Action)>>,
) -> Result<Message, Box<dyn Error + Send + Sync>> {
    let mut tokens = Vec::new();
    let mut keyboard = Vec::new();
    for row in rows {
        let mut buttons = Vec::new();
        for (label, action) in row {
            let token = match session.issue(action) {
                Ok(token) => token,
                Err(error) => {
                    session.discard(&tokens);
                    return Err(error.into());
                }
            };
            buttons.push(InlineKeyboardButton::callback(label, token.clone()));
            tokens.push(token);
        }
        keyboard.push(buttons);
    }
    let sent = bot
        .send_message(ChatId(actor.chat_id), text)
        .reply_markup(InlineKeyboardMarkup::new(keyboard))
        .await;
    match sent {
        Ok(message) => {
            session.activate(&tokens, message.id.0);
            Ok(message)
        }
        Err(error) => {
            session.discard(&tokens);
            Err(error.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::bounded;
    #[test]
    fn display_fragments_count_utf16_and_hide_controls() {
        assert_eq!(bounded("A🌸", 3), "A🌸");
        assert_eq!(bounded("A🌸B", 2), "A…");
        assert_eq!(bounded("A\0\nB", 10), "A  B");
        assert_eq!(bounded("abc", 0), "");
    }
}
